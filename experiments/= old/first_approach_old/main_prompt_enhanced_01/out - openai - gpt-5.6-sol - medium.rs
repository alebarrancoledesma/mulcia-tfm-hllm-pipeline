use std::cell::RefCell;
use std::collections::HashMap;

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::task::ExplicitFact;

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

use planforge_search::heuristics::ff::metric_operator_cost_from_initial_values;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const DEAD_END_PENALTY: f64 = 1.0e15;
const EPS: f64 = 1.0e-9;

#[derive(Clone)]
struct CustomerInfo {
    visited_fact: ExplicitFact,
    demand: f64,
}

struct LlmHeuristic {
    // Node zero is the depot. Customer i is represented by node i + 1.
    customers: Vec<CustomerInfo>,
    location_facts: Vec<ExplicitFact>,

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,

    // Directed action costs. direct[a][b] is a visit cost (or a conservative
    // fallback); via[a][b] is the visit-via-depot action cost.
    direct: Vec<Vec<f64>>,
    via: Vec<Vec<f64>>,

    // A task-dependent nearest-neighbour cycle over customers. Evaluation
    // filters visited customers from this fixed order instead of sorting.
    tour_order: Vec<usize>,
    tour_position: Vec<usize>,

    min_round_trip: f64,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_pending: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn positive_atom(name: &str, predicate: &str) -> Option<String> {
        let text = name.trim();
        let prefix = format!("Atom {}(", predicate);
        let body = text.strip_prefix(&prefix)?.strip_suffix(')')?;
        if body.is_empty() || body.contains(',') {
            None
        } else {
            Some(body.trim().to_string())
        }
    }

    fn normalise_pne_name(name: &str) -> &str {
        name.trim().strip_prefix("PNE ").unwrap_or(name.trim())
    }

    fn unary_pne_argument(name: &str, function: &str) -> Option<String> {
        let text = Self::normalise_pne_name(name);
        let prefix = format!("{}(", function);
        let body = text.strip_prefix(&prefix)?.strip_suffix(')')?;
        if body.is_empty() || body.contains(',') || body.split_whitespace().count() != 1 {
            None
        } else {
            Some(body.trim().to_string())
        }
    }

    fn is_zero_ary_pne(name: &str, alternatives: &[&str]) -> bool {
        let text = Self::normalise_pne_name(name).replace('-', "_");
        alternatives.iter().any(|candidate| {
            let expected = format!("{}()", candidate.replace('-', "_"));
            text == expected
        })
    }

    fn binary_pne_arguments(name: &str, function: &str) -> Option<(String, String)> {
        let text = Self::normalise_pne_name(name);
        let start = text.find(function)?;
        let after_name = &text[start + function.len()..];
        let body = after_name.strip_prefix('(')?.strip_suffix(')')?;

        let mut parts = body
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|part| !part.is_empty());
        let first = parts.next()?.trim().to_string();
        let second = parts.next()?.trim().to_string();
        if parts.next().is_some() {
            None
        } else {
            Some((first, second))
        }
    }

    fn operator_tokens(name: &str) -> Vec<&str> {
        name.trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split_whitespace()
            .collect()
    }

    fn finite_cost(value: f64) -> Option<f64> {
        if value.is_finite() && value >= 0.0 {
            Some(value)
        } else {
            None
        }
    }

    fn transition_cost(
        &self,
        from: usize,
        to: usize,
        load: f64,
        vehicles_left: usize,
    ) -> Option<(f64, bool)> {
        let demand = self.customers[to - 1].demand;
        if load + demand <= self.capacity + EPS {
            let cost = self.direct[from][to];
            Self::finite_cost(cost).map(|c| (c, false))
        } else if from != 0 && vehicles_left > 0 {
            let cost = self.via[from][to];
            Self::finite_cost(cost).map(|c| (c, true))
        } else {
            None
        }
    }

    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        // --------------------------------------------------------------------
        // Discover semantic propositional facts. Variable names are deliberately
        // not used because PlanForge gives them generic names such as var17.
        // --------------------------------------------------------------------
        let mut loc_by_object: HashMap<String, ExplicitFact> = HashMap::new();
        let mut visited_by_object: HashMap<String, ExplicitFact> = HashMap::new();

        for (var_id, variable) in task.variables().iter().enumerate() {
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let semantic_name = task.get_fact_name(&fact);

                if let Some(object) = Self::positive_atom(semantic_name, "loc") {
                    loc_by_object.insert(object, fact.clone());
                }
                if let Some(object) = Self::positive_atom(semantic_name, "visited") {
                    visited_by_object.insert(object, fact);
                }
            }
        }

        if visited_by_object.is_empty() {
            return Err("llm_001: no positive visited(customer) facts were found".to_string().into());
        }

        let depot_name = loc_by_object
            .keys()
            .find(|name| !visited_by_object.contains_key(*name))
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001: could not identify the depot from loc facts".to_string().into()
            })?;

        let mut customer_names: Vec<String> = visited_by_object.keys().cloned().collect();
        customer_names.sort();

        for name in &customer_names {
            if !loc_by_object.contains_key(name) {
                return Err(format!("llm_001: customer {name} has no positive loc fact").into());
            }
        }

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name.clone());
        node_names.extend(customer_names.iter().cloned());

        let node_index: HashMap<String, usize> = node_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();

        // --------------------------------------------------------------------
        // Discover dynamic numeric IDs and static resources once.
        // --------------------------------------------------------------------
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut demands: HashMap<String, f64> = HashMap::new();

        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            let value = initial_numeric[numeric_id];

            if Self::is_zero_ary_pne(name, &["load"]) {
                load_id = Some(numeric_id);
            } else if Self::is_zero_ary_pne(name, &["vehicles"]) {
                vehicles_id = Some(numeric_id);
            } else if Self::is_zero_ary_pne(name, &["capacity"]) {
                capacity = Some(value);
            } else if Self::is_zero_ary_pne(name, &["max_vehicles", "max-vehicles"]) {
                max_vehicles = Some(value);
            }

            if let Some(customer) = Self::unary_pne_argument(name, "demand") {
                if value.is_finite() && value >= 0.0 {
                    demands.insert(customer, value);
                }
            }
        }

        let load_id = load_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001: dynamic numeric variable load() was not found".to_string().into()
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001: dynamic numeric variable vehicles() was not found".to_string().into()
        })?;
        let capacity = capacity.ok_or_else(|| -> HeuristicBuildError {
            "llm_001: static capacity() value was not recoverable".to_string().into()
        })?;
        let max_vehicles = max_vehicles.ok_or_else(|| -> HeuristicBuildError {
            "llm_001: static max_vehicles() value was not recoverable".to_string().into()
        })?;

        if !capacity.is_finite() || capacity <= 0.0 || !max_vehicles.is_finite() || max_vehicles < 1.0 {
            return Err("llm_001: invalid capacity or max_vehicles value".to_string().into());
        }

        let mut customers = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let demand = demands.get(name).copied().ok_or_else(|| -> HeuristicBuildError {
                format!("llm_001: demand for customer {name} was not recoverable").into()
            })?;
            if !demand.is_finite() || demand < 0.0 {
                return Err(format!("llm_001: invalid demand for customer {name}").into());
            }
            customers.push(CustomerInfo {
                visited_fact: visited_by_object[name].clone(),
                demand,
            });
        }

        let mut location_facts = Vec::with_capacity(node_names.len());
        for name in &node_names {
            location_facts.push(loc_by_object[name].clone());
        }

        // --------------------------------------------------------------------
        // Phase A: reconstruct directed routing costs from grounded operators.
        // --------------------------------------------------------------------
        let node_count = node_names.len();
        let mut direct = vec![vec![f64::INFINITY; node_count]; node_count];
        let mut via = vec![vec![f64::INFINITY; node_count]; node_count];
        direct[0][0] = 0.0;

        for op in task.get_operators() {
            let tokens = Self::operator_tokens(op.name());
            let metric_cost = metric_operator_cost_from_initial_values(task, op);
            let Some(cost) = Self::finite_cost(metric_cost) else {
                continue;
            };

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&a), Some(&b)) = (node_index.get(*from), node_index.get(*to)) {
                        direct[a][b] = direct[a][b].min(cost);
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&a), Some(&b)) = (node_index.get(*from), node_index.get(*to)) {
                        via[a][b] = via[a][b].min(cost);
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&a) = node_index.get(*from) {
                        direct[a][0] = direct[a][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        // --------------------------------------------------------------------
        // Phase B: optional PNE fallback for only still-missing travel arcs.
        // Operator-derived values always take precedence.
        // --------------------------------------------------------------------
        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            if let Some((from, to)) = Self::binary_pne_arguments(variable.name(), "travel-cost") {
                if let (Some(&a), Some(&b)) = (node_index.get(&from), node_index.get(&to)) {
                    if !direct[a][b].is_finite() {
                        let value = initial_numeric[numeric_id];
                        if let Some(cost) = Self::finite_cost(value) {
                            direct[a][b] = cost;
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Phase C: conservative depot fallbacks. No symmetry is introduced.
        // --------------------------------------------------------------------
        for a in 1..node_count {
            for b in 1..node_count {
                if a == b {
                    direct[a][b] = 0.0;
                    continue;
                }

                if !via[a][b].is_finite()
                    && direct[a][0].is_finite()
                    && direct[0][b].is_finite()
                {
                    via[a][b] = direct[a][0] + direct[0][b];
                }

                if !direct[a][b].is_finite() {
                    if direct[a][0].is_finite() && direct[0][b].is_finite() {
                        direct[a][b] = direct[a][0] + direct[0][b];
                    } else if via[a][b].is_finite() {
                        direct[a][b] = via[a][b];
                    }
                }
            }
        }

        for customer in 1..node_count {
            if !direct[0][customer].is_finite() || !direct[customer][0].is_finite() {
                return Err(format!(
                    "llm_001: unusable routing matrix; depot legs for node {customer} are missing"
                ).into());
            }
            if customers[customer - 1].demand > capacity + EPS {
                // Construction succeeds: every state with this unvisited customer will
                // be classified as a finite-penalty resource dead end.
            }
        }

        for a in 1..node_count {
            for b in 1..node_count {
                if a != b && (!direct[a][b].is_finite() || !via[a][b].is_finite()) {
                    return Err(format!(
                        "llm_001: unusable routing matrix; arc {a}->{b} is missing"
                    ).into());
                }
            }
        }

        // --------------------------------------------------------------------
        // Precompute a directed nearest-neighbour customer cycle once.
        // --------------------------------------------------------------------
        let customer_count = customers.len();
        let mut tour_order = Vec::with_capacity(customer_count);
        let mut unused = vec![true; customer_count];
        let mut current_node = 0usize;

        for _ in 0..customer_count {
            let mut best_customer = None;
            let mut best_cost = f64::INFINITY;
            for customer in 0..customer_count {
                if unused[customer] {
                    let node = customer + 1;
                    let cost = direct[current_node][node];
                    if cost < best_cost {
                        best_cost = cost;
                        best_customer = Some(customer);
                    }
                }
            }

            let customer = best_customer.ok_or_else(|| -> HeuristicBuildError {
                "llm_001: failed to construct nearest-neighbour order".to_string().into()
            })?;
            unused[customer] = false;
            tour_order.push(customer);
            current_node = customer + 1;
        }

        let mut tour_position = vec![0usize; customer_count];
        for (position, &customer) in tour_order.iter().enumerate() {
            tour_position[customer] = position;
        }

        let min_round_trip = (1..node_count)
            .map(|node| direct[0][node] + direct[node][0])
            .filter(|cost| cost.is_finite())
            .fold(f64::INFINITY, f64::min);

        if !min_round_trip.is_finite() {
            return Err("llm_001: no finite depot round trip exists".to_string().into());
        }

        Ok(Self {
            customers,
            location_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            direct,
            via,
            tour_order,
            tour_position,
            min_round_trip,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_pending: RefCell::new(vec![false; customer_count]),
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

        let mut numeric = self.scratch_numeric.borrow_mut();
        registry.view(state).fill_numeric(&mut numeric);

        let state_load = numeric[self.load_id].max(0.0);
        let state_vehicles = numeric[self.vehicles_id].max(0.0);
        drop(numeric);

        let current_node = self
            .location_facts
            .iter()
            .position(|fact| fact.is_hold(registry.view(state)));
        let Some(current_node) = current_node else {
            return Ok(DEAD_END_PENALTY);
        };

        let mut pending = self.scratch_pending.borrow_mut();
        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;

        for (customer, info) in self.customers.iter().enumerate() {
            let unvisited = !info.visited_fact.is_hold(registry.view(state));
            pending[customer] = unvisited;
            if unvisited {
                remaining_count += 1;
                remaining_demand += info.demand;
                if info.demand > self.capacity + EPS {
                    return Ok(DEAD_END_PENALTY);
                }
            }
        }

        // All customers have been served. The only remaining domain action is
        // the final directed return-to-depot.
        if remaining_count == 0 {
            if current_node == 0 {
                return Ok(0.0);
            }
            let return_cost = self.direct[current_node][0];
            return Ok(if return_cost.is_finite() {
                return_cost
            } else {
                DEAD_END_PENALTY
            });
        }

        let used_vehicles = state_vehicles.floor().max(0.0) as usize;
        let fleet_limit = self.max_vehicles.floor().max(0.0) as usize;
        if used_vehicles > fleet_limit {
            return Ok(DEAD_END_PENALTY);
        }
        let vehicles_left = fleet_limit - used_vehicles;

        let residual_capacity = (self.capacity - state_load).max(0.0);
        let total_available_capacity = residual_capacity + vehicles_left as f64 * self.capacity;

        // Explicit resource infeasibility test required by the finite fleet.
        if remaining_demand > total_available_capacity + EPS {
            return Ok(DEAD_END_PENALTY);
        }

        // At the depot, visit-via-depot is not applicable. In valid reachable
        // states this normally means the initial state with zero load.
        if current_node == 0 && remaining_demand > 0.0 {
            let has_direct_start = self.customers.iter().enumerate().any(|(customer, info)| {
                pending[customer]
                    && state_load + info.demand <= self.capacity + EPS
                    && self.direct[0][customer + 1].is_finite()
            });
            if !has_direct_start {
                return Ok(DEAD_END_PENALTY);
            }
        }

        // ====================================================================
        // Component 1: h_routing
        //
        // Select the cheapest feasible first customer from the actual current
        // location, then filter the fixed directed nearest-neighbour cycle. The
        // simulation uses direct visits while load permits and exact via-depot
        // action costs when a new vehicle is required. It also includes the
        // final return to the depot.
        // ====================================================================
        let mut best_start = None;
        let mut best_start_cost = f64::INFINITY;

        for customer in 0..self.customers.len() {
            if !pending[customer] {
                continue;
            }
            let node = customer + 1;
            if let Some((cost, _)) = self.transition_cost(
                current_node,
                node,
                state_load,
                vehicles_left,
            ) {
                if cost < best_start_cost {
                    best_start_cost = cost;
                    best_start = Some(customer);
                }
            }
        }

        let Some(start_customer) = best_start else {
            return Ok(DEAD_END_PENALTY);
        };

        let start_position = self.tour_position[start_customer];
        let start_node = start_customer + 1;
        let start_demand = self.customers[start_customer].demand;

        let Some((mut h_routing, start_used_via)) = self.transition_cost(
            current_node,
            start_node,
            state_load,
            vehicles_left,
        ) else {
            return Ok(DEAD_END_PENALTY);
        };

        let mut simulated_vehicles_left = vehicles_left;
        let mut simulated_load;
        if start_used_via {
            simulated_vehicles_left -= 1;
            simulated_load = start_demand;
        } else {
            simulated_load = state_load + start_demand;
        }

        pending[start_customer] = false;
        let mut served = 1usize;
        let mut simulated_node = start_node;

        for offset in 1..self.tour_order.len() {
            if served == remaining_count {
                break;
            }

            let position = (start_position + offset) % self.tour_order.len();
            let customer = self.tour_order[position];
            if !pending[customer] {
                continue;
            }

            let node = customer + 1;
            let demand = self.customers[customer].demand;
            let Some((cost, used_via)) = self.transition_cost(
                simulated_node,
                node,
                simulated_load,
                simulated_vehicles_left,
            ) else {
                return Ok(DEAD_END_PENALTY);
            };

            h_routing += cost;
            if used_via {
                simulated_vehicles_left -= 1;
                simulated_load = demand;
            } else {
                simulated_load += demand;
            }
            simulated_node = node;
            pending[customer] = false;
            served += 1;
        }

        if served != remaining_count {
            return Ok(DEAD_END_PENALTY);
        }

        let final_return = self.direct[simulated_node][0];
        if !final_return.is_finite() {
            return Ok(DEAD_END_PENALTY);
        }
        h_routing += final_return;

        // ====================================================================
        // Component 2: h_capacity
        //
        // A demand-only route pressure term. It estimates how many additional
        // vehicle loads are unavoidable after exhausting current residual
        // capacity, and prices each at a fraction of the cheapest directed
        // depot round trip. This is intentionally inadmissible but improves
        // discrimination between equally geometric states with different load.
        // ====================================================================
        let excess_after_current_route = (remaining_demand - residual_capacity).max(0.0);
        let forced_new_routes = if excess_after_current_route <= EPS {
            0usize
        } else {
            (excess_after_current_route / self.capacity).ceil() as usize
        };

        if forced_new_routes > vehicles_left {
            return Ok(DEAD_END_PENALTY);
        }

        let h_capacity = 0.15 * forced_new_routes as f64 * self.min_round_trip;

        // ====================================================================
        // Component 3: h_fleet_pressure
        //
        // Prefer states that preserve fleet slack when the remaining demand
        // already requires most available vehicles.
        // ====================================================================
        let h_fleet_pressure = if forced_new_routes == 0 || vehicles_left == 0 {
            0.0
        } else {
            let tightness = forced_new_routes as f64 / vehicles_left as f64;
            0.05 * tightness * self.min_round_trip
        };

        let h = h_routing + h_capacity + h_fleet_pressure;
        if h.is_finite() && h >= 0.0 {
            Ok(h)
        } else {
            Ok(DEAD_END_PENALTY)
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
