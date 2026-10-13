use planforge_sas::numeric_task::{
    AbstractNumericTask,
    ExplicitFact,
    TaskRef,
    metric_operator_cost_from_initial_values,
};

use planforge_search::config::HeuristicSpec;

use planforge_search::evaluation::{
    EvaluationError,
    EvaluationState,
    Heuristic,
};

use planforge_search::heuristic_factory::{
    ExternalHeuristic,
    HeuristicBuildError,
    RequiredBackend,
    TaskRequirements,
};

use std::cell::RefCell;
use std::collections::HashMap;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const INF: f64 = f64::INFINITY;
const EPS: f64 = 1.0e-9;

fn parse_positive_atom(name: &str, predicate: &str) -> Option<String> {
    let text = name.trim();
    let body = text.strip_prefix("Atom ")?;
    let prefix = format!("{}(", predicate);
    let rest = body.strip_prefix(&prefix)?;
    let end = rest.find(')')?;
    let object = rest[..end].trim();
    if object.is_empty() {
        None
    } else {
        Some(object.to_string())
    }
}

fn parse_operator_name(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

fn pne_arguments<'a>(name: &'a str, function: &str) -> Option<Vec<&'a str>> {
    let text = name.trim();
    let body = text.strip_prefix("PNE ").unwrap_or(text).trim();
    let prefix = format!("{}(", function);
    let rest = body.strip_prefix(&prefix)?;
    let end = rest.find(')')?;

    // Only accept a top-level PNE for static resource extraction. This avoids
    // mistaking an opaque derived expression containing the function name for
    // the function itself.
    let suffix = rest[end + 1..].trim();
    if !suffix.is_empty() {
        return None;
    }

    let arguments = rest[..end].trim();
    if arguments.is_empty() {
        Some(Vec::new())
    } else {
        Some(arguments.split(',').map(str::trim).collect())
    }
}

fn find_zero_ary_numeric(
    task: &dyn AbstractNumericTask,
    function: &str,
) -> Option<usize> {
    task.numeric_variables()
        .iter()
        .enumerate()
        .find_map(|(id, variable)| {
            match pne_arguments(variable.name(), function) {
                Some(arguments) if arguments.is_empty() => Some(id),
                _ => None,
            }
        })
}

fn finite_cost(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

struct LlmHeuristic {
    customer_names: Vec<String>,

    // Index zero is the depot. Customer i has place index i + 1.
    loc_facts: Vec<(usize, usize)>,
    visited_facts: Vec<(usize, usize)>,

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,

    // Directed costs reconstructed primarily from grounded operators.
    direct: Vec<Vec<f64>>,
    via_depot: Vec<Vec<f64>>,
    depot_out: Vec<f64>,
    to_depot: Vec<f64>,

    // Static directed nearest-neighbour contribution for each customer.
    nearest_out_cost: Vec<f64>,

    dead_end_penalty: f64,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // --------------------------------------------------------------------
        // Discover semantic loc and visited facts. Propositional variable names
        // themselves are intentionally not used because they are generic.
        // --------------------------------------------------------------------
        let mut loc_by_object: HashMap<String, (usize, usize)> = HashMap::new();
        let mut visited_by_object: HashMap<String, (usize, usize)> = HashMap::new();

        for (var_id, variable) in task.variables().iter().enumerate() {
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(object) = parse_positive_atom(&fact_name, "loc") {
                    loc_by_object.insert(object, (var_id, value));
                }
                if let Some(object) = parse_positive_atom(&fact_name, "visited") {
                    visited_by_object.insert(object, (var_id, value));
                }
            }
        }

        if visited_by_object.is_empty() {
            return Err(
                "llm_001 could not discover any visited(customer) facts"
                    .to_string()
                    .into(),
            );
        }

        let mut customer_names: Vec<String> = visited_by_object.keys().cloned().collect();
        customer_names.sort();

        let customer_set: std::collections::HashSet<&str> =
            customer_names.iter().map(String::as_str).collect();

        let mut depot_candidates: Vec<String> = loc_by_object
            .keys()
            .filter(|name| !customer_set.contains(name.as_str()))
            .cloned()
            .collect();
        depot_candidates.sort();

        if depot_candidates.len() != 1 {
            return Err(format!(
                "llm_001 expected one depot-like loc object, found {}",
                depot_candidates.len()
            )
            .into());
        }
        let depot_name = depot_candidates.remove(0);

        let mut loc_facts = Vec::with_capacity(customer_names.len() + 1);
        loc_facts.push(*loc_by_object.get(&depot_name).ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 found a depot object without a loc fact".to_string(),
            )
        })?);

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        for customer in &customer_names {
            loc_facts.push(*loc_by_object.get(customer).ok_or_else(|| {
                HeuristicBuildError::from(format!(
                    "llm_001 found visited({customer}) but no loc({customer})"
                ))
            })?);
            visited_facts.push(*visited_by_object.get(customer).ok_or_else(|| {
                HeuristicBuildError::from(format!(
                    "llm_001 lost the visited fact for customer {customer}"
                ))
            })?);
        }

        let customer_index: HashMap<&str, usize> = customer_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();

        let mut place_index: HashMap<&str, usize> = HashMap::new();
        place_index.insert(depot_name.as_str(), 0);
        for (index, name) in customer_names.iter().enumerate() {
            place_index.insert(name.as_str(), index + 1);
        }

        // --------------------------------------------------------------------
        // Numeric resources are resolved once. They are never searched by name
        // during state evaluation.
        // --------------------------------------------------------------------
        let load_id = find_zero_ary_numeric(task, "load").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not discover the numeric fluent load()".to_string(),
            )
        })?;
        let vehicles_id = find_zero_ary_numeric(task, "vehicles").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not discover the numeric fluent vehicles()".to_string(),
            )
        })?;

        let initial_numeric = task.get_initial_numeric_state_values();

        let capacity_id = find_zero_ary_numeric(task, "capacity").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not recover capacity() from the numeric task"
                    .to_string(),
            )
        })?;
        let max_vehicles_id = find_zero_ary_numeric(task, "max_vehicles").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not recover max_vehicles() from the numeric task"
                    .to_string(),
            )
        })?;

        let capacity = *initial_numeric.get(capacity_id).ok_or_else(|| {
            HeuristicBuildError::from("llm_001 capacity id is out of range".to_string())
        })?;
        let max_vehicles = *initial_numeric.get(max_vehicles_id).ok_or_else(|| {
            HeuristicBuildError::from("llm_001 max_vehicles id is out of range".to_string())
        })?;

        if !capacity.is_finite() || capacity <= 0.0 {
            return Err("llm_001 recovered an invalid vehicle capacity"
                .to_string()
                .into());
        }
        if !max_vehicles.is_finite() || max_vehicles < 1.0 {
            return Err("llm_001 recovered an invalid fleet limit"
                .to_string()
                .into());
        }

        let mut demands = vec![INF; customer_names.len()];
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            if let Some(arguments) = pne_arguments(variable.name(), "demand") {
                if arguments.len() == 1 {
                    if let Some(&customer) = customer_index.get(arguments[0]) {
                        if let Some(&value) = initial_numeric.get(id) {
                            if value.is_finite() && value >= 0.0 {
                                demands[customer] = value;
                            }
                        }
                    }
                }
            }
        }

        if demands.iter().any(|value| !value.is_finite()) {
            return Err(
                "llm_001 could not recover every customer demand from the numeric task"
                    .to_string()
                    .into(),
            );
        }
        if demands.iter().any(|&value| value > capacity + EPS) {
            return Err(
                "llm_001 found a customer whose demand exceeds vehicle capacity"
                    .to_string()
                    .into(),
            );
        }

        // --------------------------------------------------------------------
        // Phase A: primary operator-based directed cost extraction.
        // --------------------------------------------------------------------
        let n = customer_names.len();
        let mut direct = vec![vec![INF; n]; n + 1];
        let mut via_depot = vec![vec![INF; n]; n];
        let mut to_depot = vec![INF; n + 1];
        to_depot[0] = 0.0;

        for operator in task.get_operators() {
            let tokens = parse_operator_name(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = match finite_cost(metric_operator_cost_from_initial_values(task, operator)) {
                Some(value) => value,
                None => continue,
            };

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_place), Some(&to_customer)) =
                        (place_index.get(*from), customer_index.get(*to))
                    {
                        direct[from_place][to_customer] = cost;
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_customer), Some(&to_customer)) =
                        (customer_index.get(*from), customer_index.get(*to))
                    {
                        via_depot[from_customer][to_customer] = cost;
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_customer) = customer_index.get(*from) {
                        to_depot[from_customer + 1] = cost;
                    }
                }
                _ => {}
            }
        }

        // --------------------------------------------------------------------
        // Phase B: optional PNE fallback for still-missing travel arcs.
        // --------------------------------------------------------------------
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let Some(arguments) = pne_arguments(variable.name(), "travel-cost") else {
                continue;
            };
            if arguments.len() != 2 {
                continue;
            }
            let Some(&value) = initial_numeric.get(id) else {
                continue;
            };
            let Some(cost) = finite_cost(value) else {
                continue;
            };

            let from = arguments[0];
            let to = arguments[1];
            if let (Some(&from_place), Some(&to_customer)) =
                (place_index.get(from), customer_index.get(to))
            {
                if !direct[from_place][to_customer].is_finite() {
                    direct[from_place][to_customer] = cost;
                }
            }
            if to == depot_name {
                if let Some(&from_place) = place_index.get(from) {
                    if !to_depot[from_place].is_finite() {
                        to_depot[from_place] = cost;
                    }
                }
            }
        }

        let depot_out = direct[0].clone();

        // --------------------------------------------------------------------
        // Phase C: fill only missing costs via known depot legs. No symmetry is
        // assumed: outgoing and incoming depot arcs remain separate.
        // --------------------------------------------------------------------
        for from in 0..n {
            for to in 0..n {
                if from == to {
                    continue;
                }
                if !via_depot[from][to].is_finite()
                    && to_depot[from + 1].is_finite()
                    && depot_out[to].is_finite()
                {
                    via_depot[from][to] = to_depot[from + 1] + depot_out[to];
                }
                if !direct[from + 1][to].is_finite() && via_depot[from][to].is_finite() {
                    // Conservative last resort: a legal depot-separated cost is
                    // preferable to an arbitrary invented distance.
                    direct[from + 1][to] = via_depot[from][to];
                }
            }
        }

        for customer in 0..n {
            if !depot_out[customer].is_finite() || !to_depot[customer + 1].is_finite() {
                return Err(format!(
                    "llm_001 could not reconstruct depot legs for customer {}",
                    customer_names[customer]
                )
                .into());
            }
        }

        let usable_direct_arcs = direct
            .iter()
            .flat_map(|row| row.iter())
            .filter(|value| value.is_finite())
            .count();
        if usable_direct_arcs < n {
            return Err(
                "llm_001 reconstructed an unusable directed travel-cost matrix"
                    .to_string()
                    .into(),
            );
        }

        let mut nearest_out_cost = vec![0.0; n];
        for from in 0..n {
            let mut best = to_depot[from + 1];
            for to in 0..n {
                if from != to {
                    best = best.min(direct[from + 1][to]);
                }
            }
            if !best.is_finite() {
                return Err(format!(
                    "llm_001 found no usable outgoing cost for customer {}",
                    customer_names[from]
                )
                .into());
            }
            nearest_out_cost[from] = best;
        }

        let mut max_arc: f64 = 1.0;
        for value in direct
            .iter()
            .flat_map(|row| row.iter())
            .chain(via_depot.iter().flat_map(|row| row.iter()))
            .chain(to_depot.iter())
        {
            if value.is_finite() {
                max_arc = max_arc.max(*value);
            }
        }
        let dead_end_penalty = 1.0e6 * max_arc * (n as f64 + 1.0);

        Ok(Self {
            customer_names,
            loc_facts,
            visited_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            direct,
            via_depot,
            depot_out,
            to_depot,
            nearest_out_cost,
            dead_end_penalty,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; n]),
        })
    }
}


impl Heuristic for LlmHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if eval_state.is_goal() {
            return Ok(0.0);
        }

        let _task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        let load = numeric_values
            .get(self.load_id)
            .copied()
            .unwrap_or(INF);
        let vehicles = numeric_values
            .get(self.vehicles_id)
            .copied()
            .unwrap_or(INF);

        if !load.is_finite()
            || !vehicles.is_finite()
            || load < -EPS
            || load > self.capacity + EPS
            || vehicles < 1.0 - EPS
            || vehicles > self.max_vehicles + EPS
        {
            return Ok(self.dead_end_penalty);
        }

        let mut current_place = None;
        for (place, &(var_id, value)) in self.loc_facts.iter().enumerate() {
            if propositional_values.get(var_id).copied() == Some(value) {
                current_place = Some(place);
                break;
            }
        }
        let Some(current_place) = current_place else {
            return Ok(self.dead_end_penalty);
        };

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        let mut min_return = INF;
        let mut min_depot_start = INF;
        let mut nearest_sum = 0.0;

        for customer in 0..self.customer_names.len() {
            let (var_id, value) = self.visited_facts[customer];
            let is_visited = propositional_values.get(var_id).copied() == Some(value);
            unvisited[customer] = !is_visited;

            if !is_visited {
                remaining_count += 1;
                remaining_demand += self.demands[customer];
                min_return = min_return.min(self.to_depot[customer + 1]);
                min_depot_start = min_depot_start.min(self.depot_out[customer]);
                nearest_sum += self.nearest_out_cost[customer];
            }
        }

        // Every customer has been served. The only remaining domain action is
        // the exact final return-to-depot action.
        if remaining_count == 0 {
            if current_place == 0 {
                return Ok(0.0);
            }
            let return_cost = self.to_depot[current_place];
            return if return_cost.is_finite() {
                Ok(return_cost)
            } else {
                Ok(self.dead_end_penalty)
            };
        }

        // --------------------------------------------------------------------
        // Resource infeasibility check.
        // The active vehicle has only residual capacity. Each unused vehicle
        // contributes one full capacity after a visit-via-depot transition.
        // --------------------------------------------------------------------
        let residual_capacity = (self.capacity - load).max(0.0);
        let remaining_vehicle_slots = (self.max_vehicles - vehicles).max(0.0).floor();
        let total_available_capacity =
            residual_capacity + remaining_vehicle_slots * self.capacity;

        if remaining_demand > total_available_capacity + EPS {
            return Ok(self.dead_end_penalty);
        }

        // Capacity-derived minimum number of future depot reloads. It is zero
        // when all remaining demand can fit on the currently active route.
        let demand_after_current_route =
            (remaining_demand - residual_capacity).max(0.0);
        let required_reloads = if demand_after_current_route <= EPS {
            0.0
        } else {
            (demand_after_current_route / self.capacity).ceil()
        };

        if required_reloads > remaining_vehicle_slots + EPS {
            return Ok(self.dead_end_penalty);
        }

        // --------------------------------------------------------------------
        // h_routing: state-dependent first transition + a cheap directed
        // nearest-link estimate + an explicit final return leg.
        //
        // The first transition distinguishes direct visit from visit-via-depot.
        // Both are considered when legal because the PDDL permits choosing a
        // new vehicle even when a direct visit would still fit.
        // --------------------------------------------------------------------
        let mut first_transition = INF;
        for customer in 0..self.customer_names.len() {
            if !unvisited[customer] {
                continue;
            }

            if load + self.demands[customer] <= self.capacity + EPS {
                first_transition = first_transition.min(self.direct[current_place][customer]);
            }

            if current_place > 0 && vehicles + 1.0 <= self.max_vehicles + EPS {
                first_transition = first_transition.min(
                    self.via_depot[current_place - 1][customer],
                );
            }
        }

        if !first_transition.is_finite()
            || !min_return.is_finite()
            || !min_depot_start.is_finite()
        {
            return Ok(self.dead_end_penalty);
        }

        // There are remaining_count - 1 internal customer-to-customer links.
        // Scaling the sum of static directed nearest links avoids charging one
        // internal link per customer while retaining inexpensive cluster data.
        let internal_links = if remaining_count <= 1 {
            0.0
        } else {
            nearest_sum * ((remaining_count - 1) as f64 / remaining_count as f64)
        };

        let h_routing = first_transition + internal_links + min_return;

        // --------------------------------------------------------------------
        // h_capacity: each capacity-forced new route needs a customer-to-depot
        // leg and a depot-to-customer leg. Directed minima are used separately;
        // symmetry is never assumed.
        // --------------------------------------------------------------------
        let restart_unit = min_return + min_depot_start;
        let h_capacity = required_reloads * restart_unit;

        let h = h_routing + h_capacity;
        if h.is_finite() && h >= 0.0 {
            Ok(h)
        } else {
            Ok(self.dead_end_penalty)
        }
    }

    fn heuristic_name(&self) -> &str {
        "llm_001"
    }
}


// ============================================================================
// PLANFORGE REGISTRATION
// ============================================================================

fn any_task(_: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(TaskRequirements::ANY)
}

fn no_nested_heuristics(
    _: &HeuristicSpec,
) -> Result<Vec<HeuristicSpec>, String> {
    Ok(Vec::new())
}


fn build_llm_001<'a>(
    spec: &HeuristicSpec,
    task: &'a dyn AbstractNumericTask,
    _: TaskRef<'a>,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
    if !spec.args.is_empty() {
        return Err(
            "`llm_001` does not accept arguments"
                .to_string()
                .into(),
        );
    }

    let heuristic = LlmHeuristic::new(task)?;

    Ok(Some(Box::new(heuristic)))
}


fn main() -> std::io::Result<()> {
    planforge::run_with_heuristics(vec![
        ExternalHeuristic {
            name: "llm_001",
            backend: RequiredBackend::None,
            requirements: any_task,
            nested_heuristics: no_nested_heuristics,
            build: build_llm_001,
        }
    ])
}
