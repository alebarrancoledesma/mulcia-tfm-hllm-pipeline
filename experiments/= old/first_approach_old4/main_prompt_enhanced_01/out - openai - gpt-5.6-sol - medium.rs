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
const DEAD_END_PENALTY: f64 = 1.0e12;

#[derive(Clone)]
struct CustomerInfo {
    name: String,
    place: usize,
    visited_fact: ExplicitFact,
    demand: f64,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    location_facts: Vec<ExplicitFact>,
    depot: usize,

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,

    // Directed direct-travel costs. Missing direct arcs are conservatively
    // completed through the depot during construction when possible.
    direct_cost: Vec<Vec<f64>>,

    // Exact costs of grounded visit-via-depot transitions when available.
    via_depot_cost: Vec<Vec<f64>>,

    // Cheap static continuation estimate for each customer.
    min_outgoing: Vec<f64>,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

fn extract_unary_argument(text: &str, predicate: &str) -> Option<String> {
    let marker = format!("{}(", predicate);
    let start = text.find(&marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find(')')?;
    let argument = rest[..end].trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn extract_binary_arguments(text: &str, function: &str) -> Option<(String, String)> {
    let marker = format!("{}(", function);
    let start = text.find(&marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find(')')?;
    let inside = &rest[..end];

    let parts: Vec<&str> = if inside.contains(',') {
        inside.split(',').collect()
    } else {
        inside.split_whitespace().collect()
    };

    if parts.len() != 2 {
        return None;
    }

    let first = parts[0].trim();
    let second = parts[1].trim();
    if first.is_empty() || second.is_empty() {
        None
    } else {
        Some((first.to_string(), second.to_string()))
    }
}

fn grounded_operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

fn find_zero_arity_numeric_id(
    task: &dyn AbstractNumericTask,
    function: &str,
) -> Option<usize> {
    let marker = format!("{}()", function);
    let mut best: Option<(usize, usize)> = None;

    for (id, variable) in task.numeric_variables().iter().enumerate() {
        let name = variable.name();
        if name.contains(&marker) {
            // Prefer the shortest name. This normally selects the original
            // zero-arity PNE rather than a longer compiled expression name.
            let score = name.len();
            if best.map_or(true, |(_, old_score)| score < old_score) {
                best = Some((id, score));
            }
        }
    }

    best.map(|(id, _)| id)
}

fn find_static_numeric_value(
    task: &dyn AbstractNumericTask,
    function: &str,
) -> Option<f64> {
    let marker = format!("{}()", function);
    let initial = task.get_initial_numeric_state_values();
    let mut best: Option<(f64, usize)> = None;

    for (id, variable) in task.numeric_variables().iter().enumerate() {
        let name = variable.name();
        if name.contains(&marker) && id < initial.len() {
            let value = initial[id];
            if value.is_finite() {
                let score = name.len();
                if best.map_or(true, |(_, old_score)| score < old_score) {
                    best = Some((value, score));
                }
            }
        }
    }

    best.map(|(value, _)| value)
}

fn find_customer_demand(
    task: &dyn AbstractNumericTask,
    customer: &str,
) -> Option<f64> {
    let initial = task.get_initial_numeric_state_values();
    let mut best: Option<(f64, usize)> = None;

    for (id, variable) in task.numeric_variables().iter().enumerate() {
        if id >= initial.len() {
            continue;
        }

        let name = variable.name();
        if let Some(argument) = extract_unary_argument(name, "demand") {
            if argument == customer {
                let value = initial[id];
                if value.is_finite() && value >= 0.0 {
                    let score = name.len();
                    if best.map_or(true, |(_, old_score)| score < old_score) {
                        best = Some((value, score));
                    }
                }
            }
        }
    }

    best.map(|(value, _)| value)
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Phase 1: discover semantic loc and visited facts. Propositional
        // variable names themselves are intentionally not used.
        let mut place_names: Vec<String> = Vec::new();
        let mut location_facts: Vec<ExplicitFact> = Vec::new();
        let mut place_index: HashMap<String, usize> = HashMap::new();
        let mut visited_by_name: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(place_name) = extract_unary_argument(&fact_name, "loc") {
                    if !place_index.contains_key(&place_name) {
                        let index = place_names.len();
                        place_index.insert(place_name.clone(), index);
                        place_names.push(place_name);
                        location_facts.push(fact.clone());
                    }
                }

                if let Some(customer_name) = extract_unary_argument(&fact_name, "visited") {
                    visited_by_name.entry(customer_name).or_insert(fact);
                }
            }
        }

        if place_names.is_empty() || visited_by_name.is_empty() {
            return Err(
                "llm_001 could not discover CVRP loc/visited facts from semantic fact names"
                    .to_string()
                    .into(),
            );
        }

        // The depot is the place having a loc fact but no corresponding
        // visited fact. The supplied domain has exactly one such place.
        let depot_candidates: Vec<usize> = place_names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                if visited_by_name.contains_key(name) {
                    None
                } else {
                    Some(index)
                }
            })
            .collect();

        if depot_candidates.len() != 1 {
            return Err(
                "llm_001 expected exactly one depot identifiable from loc/visited facts"
                    .to_string()
                    .into(),
            );
        }
        let depot = depot_candidates[0];

        let load_id = find_zero_arity_numeric_id(task, "load").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not identify the numeric load() fluent".to_string(),
            )
        })?;
        let vehicles_id = find_zero_arity_numeric_id(task, "vehicles").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not identify the numeric vehicles() fluent".to_string(),
            )
        })?;

        let capacity = find_static_numeric_value(task, "capacity").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not recover the CVRP capacity constant".to_string(),
            )
        })?;
        let max_vehicles = find_static_numeric_value(task, "max_vehicles").ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not recover the CVRP max_vehicles constant".to_string(),
            )
        })?;

        if !capacity.is_finite() || capacity <= 0.0 ||
           !max_vehicles.is_finite() || max_vehicles < 1.0 {
            return Err(
                "llm_001 recovered invalid capacity or max_vehicles values"
                    .to_string()
                    .into(),
            );
        }

        let mut customers = Vec::with_capacity(visited_by_name.len());
        for (place, name) in place_names.iter().enumerate() {
            if let Some(visited_fact) = visited_by_name.get(name) {
                let demand = find_customer_demand(task, name).ok_or_else(|| {
                    HeuristicBuildError::from(format!(
                        "llm_001 could not recover demand for customer {}",
                        name
                    ))
                })?;

                if demand > capacity + 1.0e-9 {
                    return Err(
                        format!(
                            "llm_001 found customer demand exceeding vehicle capacity for {}",
                            name
                        )
                        .into(),
                    );
                }

                customers.push(CustomerInfo {
                    name: name.clone(),
                    place,
                    visited_fact: visited_fact.clone(),
                    demand,
                });
            }
        }

        let place_count = place_names.len();
        let mut direct_cost = vec![vec![INF; place_count]; place_count];
        let mut via_depot_cost = vec![vec![INF; place_count]; place_count];
        for i in 0..place_count {
            direct_cost[i][i] = 0.0;
        }

        // Phase A: primary extraction from grounded operators. These costs
        // are the metric costs evaluated from the task's initial constants.
        for operator in task.get_operators() {
            let tokens = grounded_operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        direct_cost[from][to] = direct_cost[from][to].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        via_depot_cost[from][to] = via_depot_cost[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = place_index.get(tokens[1]) {
                        direct_cost[from][depot] =
                            direct_cost[from][depot].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Phase B: optional PNE fallback. It only fills arcs still missing
        // after operator-based extraction and never overrides operator costs.
        let initial_numeric = task.get_initial_numeric_state_values();
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            if id >= initial_numeric.len() {
                continue;
            }
            if let Some((from_name, to_name)) =
                extract_binary_arguments(variable.name(), "travel-cost")
            {
                if let (Some(&from), Some(&to)) =
                    (place_index.get(&from_name), place_index.get(&to_name))
                {
                    let value = initial_numeric[id];
                    if !direct_cost[from][to].is_finite() &&
                       value.is_finite() && value >= 0.0
                    {
                        direct_cost[from][to] = value;
                    }
                }
            }
        }

        // Phase C: complete only missing arcs via known directed depot legs.
        // No symmetry is assumed or introduced.
        for from in 0..place_count {
            for to in 0..place_count {
                if !direct_cost[from][to].is_finite() &&
                   direct_cost[from][depot].is_finite() &&
                   direct_cost[depot][to].is_finite()
                {
                    direct_cost[from][to] =
                        direct_cost[from][depot] + direct_cost[depot][to];
                }

                if !via_depot_cost[from][to].is_finite() &&
                   direct_cost[from][depot].is_finite() &&
                   direct_cost[depot][to].is_finite()
                {
                    via_depot_cost[from][to] =
                        direct_cost[from][depot] + direct_cost[depot][to];
                }
            }
        }

        // Every customer must be reachable from the depot and able to return.
        for customer in &customers {
            if !direct_cost[depot][customer.place].is_finite() ||
               !direct_cost[customer.place][depot].is_finite()
            {
                return Err(
                    format!(
                        "llm_001 has an unusable directed cost matrix around customer {}",
                        customer.name
                    )
                    .into(),
                );
            }
        }

        // Precompute a cheap geometric continuation term. It is directed and
        // therefore does not assume symmetric travel costs.
        let mut min_outgoing = vec![0.0; customers.len()];
        for (i, customer) in customers.iter().enumerate() {
            let mut best = direct_cost[customer.place][depot];
            for other in &customers {
                if other.place != customer.place {
                    best = best.min(direct_cost[customer.place][other.place]);
                }
            }
            if !best.is_finite() {
                return Err(
                    format!(
                        "llm_001 found no usable outgoing arc for customer {}",
                        customer.name
                    )
                    .into(),
                );
            }
            min_outgoing[i] = best;
        }

        Ok(Self {
            customers,
            location_facts,
            depot,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            direct_cost,
            via_depot_cost,
            min_outgoing,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; visited_by_name.len()]),
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

        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let state_view = registry.view(state);

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        state_view.fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        // Keep a reusable propositional buffer populated as well. Semantic
        // fact truth is tested through ExplicitFact::is_hold so no assumption
        // is made about which finite-domain value denotes truth.
        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        if self.load_id >= numeric_values.len() ||
           self.vehicles_id >= numeric_values.len()
        {
            return Err(EvaluationError::ComputationFailed(
                "llm_001 numeric state is shorter than its cached fluent IDs".to_string(),
            ));
        }

        let load = numeric_values[self.load_id].max(0.0);
        let vehicles = numeric_values[self.vehicles_id].max(0.0);
        drop(numeric_values);
        drop(propositional_values);

        let current_place = self.location_facts
            .iter()
            .position(|fact| fact.is_hold(state_view));

        let current_place = match current_place {
            Some(place) => place,
            None => return Ok(DEAD_END_PENALTY),
        };

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        if unvisited.len() != self.customers.len() {
            unvisited.resize(self.customers.len(), false);
        }

        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        for (i, customer) in self.customers.iter().enumerate() {
            let pending = !customer.visited_fact.is_hold(state_view);
            unvisited[i] = pending;
            if pending {
                remaining_count += 1;
                remaining_demand += customer.demand;
            }
        }

        // All customers are served: the only remaining action is the exact
        // directed return-to-depot action, unless already at the depot.
        if remaining_count == 0 {
            if current_place == self.depot {
                return Ok(0.0);
            }
            let return_cost = self.direct_cost[current_place][self.depot];
            return if return_cost.is_finite() {
                Ok(return_cost.max(0.0))
            } else {
                Ok(DEAD_END_PENALTY)
            };
        }

        let vehicles_left = (self.max_vehicles - vehicles).max(0.0).floor();
        let residual_capacity = (self.capacity - load).max(0.0);
        let total_available_capacity =
            residual_capacity + vehicles_left * self.capacity;

        // Resource dead-end detection mandated by the fleet-limited CVRP.
        if remaining_demand > total_available_capacity + 1.0e-9 {
            let excess = remaining_demand - total_available_capacity;
            return Ok(DEAD_END_PENALTY + excess * 1.0e6 + remaining_count as f64);
        }

        // Component 1: directed routing estimate.
        //
        // The first transition is modeled with the actual domain distinction:
        // a direct visit is considered only if the next demand fits, while a
        // visit-via-depot transition is considered only if another vehicle is
        // available. At the depot, visits start directly.
        let mut first_transition = INF;
        let mut min_depot_open = INF;
        let mut min_depot_close = INF;
        let mut continuation_sum = 0.0;

        for (i, customer) in self.customers.iter().enumerate() {
            if !unvisited[i] {
                continue;
            }

            min_depot_open = min_depot_open.min(
                self.direct_cost[self.depot][customer.place]
            );
            min_depot_close = min_depot_close.min(
                self.direct_cost[customer.place][self.depot]
            );
            continuation_sum += self.min_outgoing[i];

            if current_place == self.depot {
                if load + customer.demand <= self.capacity + 1.0e-9 {
                    first_transition = first_transition.min(
                        self.direct_cost[self.depot][customer.place]
                    );
                }
            } else {
                if load + customer.demand <= self.capacity + 1.0e-9 {
                    first_transition = first_transition.min(
                        self.direct_cost[current_place][customer.place]
                    );
                }

                if vehicles_left >= 1.0 {
                    first_transition = first_transition.min(
                        self.via_depot_cost[current_place][customer.place]
                    );
                }
            }
        }

        if !first_transition.is_finite() ||
           !min_depot_open.is_finite() ||
           !min_depot_close.is_finite()
        {
            return Ok(DEAD_END_PENALTY + remaining_count as f64);
        }

        // The explicit closing term ensures that the final return to the depot
        // is represented whenever customers remain. The continuation term
        // captures geographic dispersion without an O(n^2) state computation.
        let h_routing = first_transition
            + 0.65 * continuation_sum
            + min_depot_close;

        // Component 2: capacity/fleet route-bound estimate.
        // Estimate how many additional depot reloads are unavoidable after
        // consuming the residual capacity of the current vehicle.
        let demand_after_current_vehicle =
            (remaining_demand - residual_capacity).max(0.0);
        let additional_routes = if demand_after_current_vehicle <= 1.0e-9 {
            0.0
        } else {
            (demand_after_current_vehicle / self.capacity).ceil()
        };

        if additional_routes > vehicles_left + 1.0e-9 {
            return Ok(DEAD_END_PENALTY + remaining_count as f64);
        }

        // Each additional route has a depot opening and closing burden. A
        // half-weight avoids overwhelming the geometric term while still
        // differentiating states with different load and vehicle usage.
        let reload_burden = 0.5 * (min_depot_open + min_depot_close);
        let h_capacity = additional_routes * reload_burden;

        let h = h_routing + h_capacity;
        if h.is_finite() && h >= 0.0 {
            Ok(h)
        } else {
            Ok(DEAD_END_PENALTY + remaining_count as f64)
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
                .into()
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
