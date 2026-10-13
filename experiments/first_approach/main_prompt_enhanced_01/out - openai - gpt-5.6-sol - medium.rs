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

const LARGE_PENALTY: f64 = 1.0e15;
const INF: f64 = f64::INFINITY;

#[derive(Clone, Copy)]
struct FactIndex {
    var_id: usize,
    value: usize,
}

struct LlmHeuristic {
    customer_count: usize,
    depot_node: usize,
    loc_facts: Vec<FactIndex>,
    visited_facts: Vec<FactIndex>,

    load_id: usize,
    vehicles_id: usize,

    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,

    // Node 0 is the depot and nodes 1..=customer_count are customers.
    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,
    min_customer_out: Vec<f64>,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

fn normalized_pne_name(name: &str) -> &str {
    let trimmed = name.trim();
    if let Some(rest) = trimmed.strip_prefix("PNE ") {
        rest.trim()
    } else {
        trimmed
    }
}

fn parse_unary_atom(name: &str, predicate: &str) -> Option<String> {
    let mut text = name.trim();
    if let Some(rest) = text.strip_prefix("Atom ") {
        text = rest.trim();
    }

    let prefix = format!("{}(", predicate);
    if !text.starts_with(&prefix) || !text.ends_with(')') {
        return None;
    }

    let argument = text[prefix.len()..text.len() - 1].trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn parse_binary_function(name: &str, function: &str) -> Option<(String, String)> {
    let text = normalized_pne_name(name);
    let prefix = format!("{}(", function);
    if !text.starts_with(&prefix) || !text.ends_with(')') {
        return None;
    }

    let arguments = text[prefix.len()..text.len() - 1].trim();
    let parts: Vec<&str> = if arguments.contains(',') {
        arguments.split(',').map(str::trim).collect()
    } else {
        arguments.split_whitespace().collect()
    };

    if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover semantic propositional facts. Propositional variable names are
        // deliberately not used because they are generic after translation.
        let mut loc_by_place: HashMap<String, FactIndex> = HashMap::new();
        let mut visited_by_customer: HashMap<String, FactIndex> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(place) = parse_unary_atom(fact_name, "loc") {
                    loc_by_place.insert(place, FactIndex { var_id, value });
                }
                if let Some(customer) = parse_unary_atom(fact_name, "visited") {
                    visited_by_customer.insert(customer, FactIndex { var_id, value });
                }
            }
        }

        if visited_by_customer.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not discover any visited(customer) facts".to_string(),
            ));
        }

        // A depot has a loc fact but no visited fact. The target family has one depot.
        let depot_candidates: Vec<String> = loc_by_place
            .keys()
            .filter(|name| !visited_by_customer.contains_key(*name))
            .cloned()
            .collect();

        if depot_candidates.len() != 1 {
            return Err(HeuristicBuildError::from(format!(
                "llm_001 expected one loc-only depot, found {}",
                depot_candidates.len()
            )));
        }
        let depot_name = depot_candidates[0].clone();

        let mut customer_names: Vec<String> = visited_by_customer.keys().cloned().collect();
        customer_names.sort();
        let customer_count = customer_names.len();
        let depot_node = 0usize;
        let node_count = customer_count + 1;

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), depot_node);
        for (customer_id, name) in customer_names.iter().enumerate() {
            node_by_name.insert(name.clone(), customer_id + 1);
        }

        let depot_loc = *loc_by_place.get(&depot_name).ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 found a depot name but no depot loc fact"
                    .to_string()
                    .into()
            },
        )?;

        let mut loc_facts = vec![depot_loc; node_count];
        let mut visited_facts = Vec::with_capacity(customer_count);
        for (customer_id, name) in customer_names.iter().enumerate() {
            loc_facts[customer_id + 1] = *loc_by_place.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    format!("llm_001 found visited({name}) but no loc({name}) fact").into()
                },
            )?;
            visited_facts.push(*visited_by_customer.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    format!("llm_001 lost the visited fact for {name}").into()
                },
            )?);
        }

        // Build an exact-name numeric map once. No substring matching is used for
        // load, vehicles, capacity, max_vehicles, or demand.
        let mut numeric_by_exact_name: HashMap<String, usize> = HashMap::new();
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            numeric_by_exact_name.insert(
                normalized_pne_name(variable.name()).to_string(),
                id,
            );
        }

        let load_id = *numeric_by_exact_name.get("load()").ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 could not find the exact numeric fluent load()"
                    .to_string()
                    .into()
            },
        )?;
        let vehicles_id = *numeric_by_exact_name.get("vehicles()").ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 could not find the exact numeric fluent vehicles()"
                    .to_string()
                    .into()
            },
        )?;

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = numeric_by_exact_name
            .get("capacity()")
            .and_then(|id| initial_numeric.get(*id))
            .copied()
            .filter(|value| finite_nonnegative(*value))
            .unwrap_or(f64::MAX);
        let max_vehicles = numeric_by_exact_name
            .get("max_vehicles()")
            .and_then(|id| initial_numeric.get(*id))
            .copied()
            .filter(|value| finite_nonnegative(*value))
            .unwrap_or(f64::MAX);

        let mut demands = vec![0.0_f64; customer_count];
        for (customer_id, customer_name) in customer_names.iter().enumerate() {
            let exact_name = format!("demand({customer_name})");
            if let Some(id) = numeric_by_exact_name.get(&exact_name) {
                if let Some(value) = initial_numeric.get(*id) {
                    if finite_nonnegative(*value) {
                        demands[customer_id] = *value;
                    }
                }
            }
        }

        let mut direct_cost = vec![vec![INF; node_count]; node_count];
        let mut via_depot_cost = vec![vec![INF; node_count]; node_count];
        let mut return_cost = vec![INF; node_count];
        direct_cost[depot_node][depot_node] = 0.0;
        return_cost[depot_node] = 0.0;

        // Phase A: grounded operators are the primary source of routing costs.
        for operator in task.get_operators() {
            let tokens: Vec<&str> = operator.name().split_whitespace().collect();
            let metric_cost = metric_operator_cost_from_initial_values(task, operator);
            if !finite_nonnegative(metric_cost) {
                continue;
            }

            if tokens.len() == 3 && tokens[0] == "visit" {
                if let (Some(&from), Some(&to)) =
                    (node_by_name.get(tokens[1]), node_by_name.get(tokens[2]))
                {
                    direct_cost[from][to] = direct_cost[from][to].min(metric_cost);
                }
            } else if tokens.len() == 3 && tokens[0] == "visit-via-depot" {
                if let (Some(&from), Some(&to)) =
                    (node_by_name.get(tokens[1]), node_by_name.get(tokens[2]))
                {
                    via_depot_cost[from][to] =
                        via_depot_cost[from][to].min(metric_cost);
                }
            } else if tokens.len() == 2 && tokens[0] == "return-to-depot" {
                if let Some(&from) = node_by_name.get(tokens[1]) {
                    return_cost[from] = return_cost[from].min(metric_cost);
                    direct_cost[from][depot_node] =
                        direct_cost[from][depot_node].min(metric_cost);
                }
            }
        }

        // Phase B: exact parsing of travel-cost(from,to) PNEs fills only gaps.
        // This is a fallback; it is not the primary source of costs.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            if let Some((from_name, to_name)) =
                parse_binary_function(variable.name(), "travel-cost")
            {
                if let (Some(&from), Some(&to), Some(&value)) = (
                    node_by_name.get(&from_name),
                    node_by_name.get(&to_name),
                    initial_numeric.get(id),
                ) {
                    if !direct_cost[from][to].is_finite() && finite_nonnegative(value) {
                        direct_cost[from][to] = value;
                    }
                    if to == depot_node
                        && !return_cost[from].is_finite()
                        && finite_nonnegative(value)
                    {
                        return_cost[from] = value;
                    }
                }
            }
        }

        // Phase C: derive via-depot transitions from known directed depot legs.
        // Missing direct customer arcs are conservatively represented by this
        // directed depot path. No symmetry assumption is made.
        for from in 1..node_count {
            if !return_cost[from].is_finite() && direct_cost[from][depot_node].is_finite() {
                return_cost[from] = direct_cost[from][depot_node];
            }
            for to in 1..node_count {
                if from == to {
                    continue;
                }
                if !via_depot_cost[from][to].is_finite()
                    && return_cost[from].is_finite()
                    && direct_cost[depot_node][to].is_finite()
                {
                    via_depot_cost[from][to] =
                        return_cost[from] + direct_cost[depot_node][to];
                }
                if !direct_cost[from][to].is_finite()
                    && via_depot_cost[from][to].is_finite()
                {
                    direct_cost[from][to] = via_depot_cost[from][to];
                }
            }
        }

        // Every customer needs a depot departure and a final directed return.
        for node in 1..node_count {
            if !direct_cost[depot_node][node].is_finite()
                || !return_cost[node].is_finite()
            {
                return Err(HeuristicBuildError::from(format!(
                    "llm_001 could not reconstruct usable directed depot legs for customer node {}",
                    node
                )));
            }
        }

        // Precompute a relaxed cheapest outgoing customer edge. At evaluation,
        // summing all but the largest of these gives a cheap path-forest proxy.
        let mut min_customer_out = vec![0.0_f64; node_count];
        for from in 1..node_count {
            let mut best: f64 = INF;
            for to in 1..node_count {
                if from != to {
                    best = best.min(direct_cost[from][to]);
                }
            }
            min_customer_out[from] = if best.is_finite() { best } else { 0.0 };
        }

        Ok(Self {
            customer_count,
            depot_node,
            loc_facts,
            visited_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            direct_cost,
            via_depot_cost,
            return_cost,
            min_customer_out,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; customer_count]),
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

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        let load = numeric_values
            .get(self.load_id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let vehicles = numeric_values
            .get(self.vehicles_id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);

        let mut current_node: Option<usize> = None;
        for (node, fact) in self.loc_facts.iter().enumerate() {
            if propositional_values.get(fact.var_id).copied() == Some(fact.value) {
                current_node = Some(node);
                break;
            }
        }
        let current_node = current_node.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "llm_001 could not determine the current loc fact".to_string(),
            )
        })?;

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        let mut remaining_count = 0usize;
        let mut remaining_demand: f64 = 0.0;
        let mut largest_remaining_demand: f64 = 0.0;

        for customer_id in 0..self.customer_count {
            let fact = self.visited_facts[customer_id];
            let is_visited =
                propositional_values.get(fact.var_id).copied() == Some(fact.value);
            unvisited[customer_id] = !is_visited;
            if !is_visited {
                remaining_count += 1;
                remaining_demand += self.demands[customer_id];
                largest_remaining_demand =
                    largest_remaining_demand.max(self.demands[customer_id]);
            }
        }

        // All customers are served: only the mandatory final return remains.
        if remaining_count == 0 {
            if current_node == self.depot_node {
                return Ok(0.0);
            }
            let return_leg = self.return_cost[current_node];
            return if return_leg.is_finite() {
                Ok(return_leg.max(0.0))
            } else {
                Ok(LARGE_PENALTY)
            };
        }

        let capacity_known = self.capacity < f64::MAX / 2.0;
        let fleet_known = self.max_vehicles < f64::MAX / 2.0;

        // Best-effort resource dead-end detection. Missing demand values are zero,
        // while missing capacity/fleet values are permissively unbounded.
        if capacity_known && largest_remaining_demand > self.capacity + 1.0e-9 {
            return Ok(LARGE_PENALTY);
        }

        if capacity_known && fleet_known {
            let residual_capacity = (self.capacity - load).max(0.0);
            let unused_vehicles = (self.max_vehicles - vehicles).max(0.0);
            let total_available = residual_capacity + unused_vehicles * self.capacity;
            if remaining_demand > total_available + 1.0e-9 {
                return Ok(LARGE_PENALTY);
            }
        }

        // Find the cheapest legal first transition. From the depot only a direct
        // visit is available. From a customer, a fitting direct visit and a
        // visit-via-depot are considered separately according to their semantics.
        let can_start_new_vehicle = !fleet_known || vehicles + 1.0 <= self.max_vehicles + 1.0e-9;
        let mut first_transition: f64 = INF;
        let mut any_direct_fit = false;
        let mut min_final_return: f64 = INF;
        let mut min_depot_departure: f64 = INF;
        let mut relaxed_sum: f64 = 0.0;
        let mut relaxed_largest: f64 = 0.0;

        for customer_id in 0..self.customer_count {
            if !unvisited[customer_id] {
                continue;
            }

            let node = customer_id + 1;
            let demand = self.demands[customer_id];
            let fits = !capacity_known || load + demand <= self.capacity + 1.0e-9;

            min_final_return = min_final_return.min(self.return_cost[node]);
            min_depot_departure =
                min_depot_departure.min(self.direct_cost[self.depot_node][node]);

            let relaxed_edge = self.min_customer_out[node];
            relaxed_sum += relaxed_edge;
            relaxed_largest = relaxed_largest.max(relaxed_edge);

            if current_node == self.depot_node {
                if fits {
                    let cost = self.direct_cost[self.depot_node][node];
                    if cost.is_finite() {
                        any_direct_fit = true;
                        first_transition = first_transition.min(cost);
                    }
                }
            } else {
                if fits {
                    let cost = self.direct_cost[current_node][node];
                    if cost.is_finite() {
                        any_direct_fit = true;
                        first_transition = first_transition.min(cost);
                    }
                }

                if can_start_new_vehicle {
                    let cost = self.via_depot_cost[current_node][node];
                    if cost.is_finite() {
                        first_transition = first_transition.min(cost);
                    }
                }
            }
        }

        // At the depot, visit-via-depot is not applicable. At a customer, if no
        // direct customer fits and no new vehicle is available, no move remains.
        if current_node == self.depot_node && !any_direct_fit {
            return Ok(LARGE_PENALTY);
        }
        if current_node != self.depot_node
            && !any_direct_fit
            && !can_start_new_vehicle
        {
            return Ok(LARGE_PENALTY);
        }
        if !first_transition.is_finite()
            || !min_final_return.is_finite()
            || !min_depot_departure.is_finite()
        {
            return Ok(LARGE_PENALTY);
        }

        // Component 1: directed geometric routing proxy.
        // It contains the state-specific first legal transition, a relaxed set of
        // interior customer edges, and an explicit final return to the depot.
        let relaxed_interior = if remaining_count > 1 {
            (relaxed_sum - relaxed_largest).max(0.0)
        } else {
            0.0
        };
        let h_routing = first_transition + relaxed_interior + min_final_return;

        // Estimate how many additional depot reloads are forced by aggregate
        // demand after using the residual capacity of the current vehicle.
        let mut required_reloads: f64 = 0.0;
        if capacity_known && self.capacity > 0.0 {
            let residual = (self.capacity - load).max(0.0);
            let demand_after_current = (remaining_demand - residual).max(0.0);
            required_reloads = (demand_after_current / self.capacity).ceil();
        }

        let depot_pair_cost = (min_final_return + min_depot_departure).max(0.0);

        // Component 2: capacity-induced route-break cost.
        let h_capacity = required_reloads * depot_pair_cost;

        // Component 3: fleet/load pressure. With known limits this increases as
        // the estimated reload count consumes the remaining fleet. If constants
        // were folded away, a small bounded pressure term still lets dynamic load
        // and vehicles distinguish otherwise geometrically identical states.
        let h_vehicles = if fleet_known {
            let remaining_slots = (self.max_vehicles - vehicles).max(0.0);
            if required_reloads > 0.0 {
                0.05 * depot_pair_cost * required_reloads / (remaining_slots + 1.0)
            } else {
                0.0
            }
        } else {
            let vehicle_pressure = vehicles / (vehicles + 1.0);
            let load_pressure = load / (load + 1.0);
            0.001 * depot_pair_cost * (vehicle_pressure + load_pressure)
        };

        let h = (h_routing + h_capacity + h_vehicles).max(0.0);
        if h.is_finite() {
            Ok(h)
        } else {
            Ok(LARGE_PENALTY)
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
