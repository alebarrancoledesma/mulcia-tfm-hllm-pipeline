use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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

const INF: f64 = f64::INFINITY;

#[derive(Clone)]
struct CustomerInfo {
    visited_fact: ExplicitFact,
    loc_fact: ExplicitFact,
    demand: f64,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    depot_loc_fact: ExplicitFact,
    depot_index: usize,

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,

    // Directed direct-visit costs. The last row/column represents the depot.
    direct_cost: Vec<Vec<f64>>,
    // Directed costs of the domain's visit-via-depot action.
    via_depot_cost: Vec<Vec<f64>>,
    return_to_depot: Vec<f64>,
    min_incoming: Vec<f64>,

    dead_end_penalty: f64,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_remaining: RefCell<Vec<bool>>,
}

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let needle = format!("{}(", predicate);
    let start = name.find(&needle)? + needle.len();
    let rest = &name[start..];
    let end = rest.find(')')?;
    let arg = rest[..end].trim();
    if arg.is_empty() {
        None
    } else {
        Some(arg.to_string())
    }
}

fn function_arguments(name: &str, function: &str) -> Option<Vec<String>> {
    let needle = format!("{}(", function);
    let start = name.find(&needle)? + needle.len();
    let rest = &name[start..];
    let end = rest.find(')')?;
    let inside = &rest[..end];
    Some(
        inside
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    )
}

fn normalized_numeric_name(name: &str) -> String {
    name.to_ascii_lowercase()
        .replace([' ', '_', '-'], "")
}

fn is_zero_ary_numeric(name: &str, function: &str) -> bool {
    let normalized = normalized_numeric_name(name);
    let wanted = format!("{}()", normalized_numeric_name(function));
    normalized.contains(&wanted)
}

fn ceil_nonnegative(value: f64) -> f64 {
    if value <= 1.0e-9 {
        0.0
    } else {
        value.ceil()
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let operators = task.get_operators();

        // First recover object names from grounded routing operators. Customer
        // identity is structural: every destination of visit is a customer.
        let mut customer_names = Vec::<String>::new();
        let mut customer_seen = HashSet::<String>::new();
        let mut visit_records = Vec::<(String, String, f64)>::new();
        let mut via_records = Vec::<(String, String, f64)>::new();
        let mut return_records = Vec::<(String, f64)>::new();
        let mut direct_sources = HashSet::<String>::new();

        for op in operators.iter() {
            let parts: Vec<&str> = op.name().split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "visit" {
                let from = parts[1].to_string();
                let to = parts[2].to_string();
                let cost = metric_operator_cost_from_initial_values(task, op);
                if cost.is_finite() && cost >= 0.0 {
                    visit_records.push((from.clone(), to.clone(), cost));
                }
                direct_sources.insert(from);
                if customer_seen.insert(to.clone()) {
                    customer_names.push(to);
                }
            } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
                let from = parts[1].to_string();
                let to = parts[2].to_string();
                let cost = metric_operator_cost_from_initial_values(task, op);
                if cost.is_finite() && cost >= 0.0 {
                    via_records.push((from, to.clone(), cost));
                }
                if customer_seen.insert(to.clone()) {
                    customer_names.push(to);
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                let from = parts[1].to_string();
                let cost = metric_operator_cost_from_initial_values(task, op);
                if cost.is_finite() && cost >= 0.0 {
                    return_records.push((from, cost));
                }
            }
        }

        if customer_names.is_empty() {
            return Err(
                "llm_001 could not find grounded CVRP visit operators"
                    .to_string()
                    .into(),
            );
        }

        // In this domain family there is one location that is a direct-visit
        // source but never a visit destination: the depot.
        let depot_name = direct_sources
            .iter()
            .find(|name| !customer_seen.contains(*name))
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not identify the unique depot structurally"
                    .to_string()
                    .into()
            })?;

        let mut customer_index = HashMap::<String, usize>::new();
        for (index, name) in customer_names.iter().enumerate() {
            customer_index.insert(name.clone(), index);
        }
        let depot_index = customer_names.len();
        let place_count = depot_index + 1;

        // Mandatory semantic fact-name scan. Generic SAS variable names are
        // deliberately ignored. Goals provide every visited fact and the depot
        // location; grounded preconditions provide all customer locations.
        let mut visited_facts = HashMap::<String, ExplicitFact>::new();
        let mut location_facts = HashMap::<String, ExplicitFact>::new();

        for goal_index in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_index);
            let semantic_name = task.get_fact_name(fact);
            if let Some(object) = atom_argument(semantic_name, "visited") {
                visited_facts.insert(object, (*fact).clone());
            }
            if let Some(object) = atom_argument(semantic_name, "loc") {
                location_facts.insert(object, (*fact).clone());
            }
        }

        for op in operators.iter() {
            for fact in op.preconditions().iter() {
                let semantic_name = task.get_fact_name(fact);
                if let Some(object) = atom_argument(semantic_name, "visited") {
                    // Only positive Atom facts are useful here. A negated fact
                    // name is not inserted as the visited marker.
                    if semantic_name.starts_with("Atom ") {
                        visited_facts.entry(object).or_insert_with(|| fact.clone());
                    }
                }
                if let Some(object) = atom_argument(semantic_name, "loc") {
                    if semantic_name.starts_with("Atom ") {
                        location_facts.entry(object).or_insert_with(|| fact.clone());
                    }
                }
            }
        }

        let depot_loc_fact = location_facts
            .get(&depot_name)
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001 found the depot but not its semantic loc fact"
                    .to_string()
                    .into()
            })?;

        // Discover dynamic numeric IDs once. Static values and optional PNE
        // fallback data are read from the initial numeric vector.
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut demands = HashMap::<String, f64>::new();
        let mut fallback_travel = Vec::<(String, String, f64)>::new();

        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            let value = initial_numeric[numeric_id];

            if is_zero_ary_numeric(name, "load") {
                load_id = Some(numeric_id);
            } else if is_zero_ary_numeric(name, "vehicles") {
                vehicles_id = Some(numeric_id);
            }

            if is_zero_ary_numeric(name, "capacity") && value.is_finite() {
                capacity = Some(value);
            }
            if is_zero_ary_numeric(name, "max_vehicles") && value.is_finite() {
                max_vehicles = Some(value);
            }

            if let Some(args) = function_arguments(name, "demand") {
                if args.len() == 1 && value.is_finite() && value >= 0.0 {
                    demands.insert(args[0].clone(), value);
                }
            }

            // Phase B fallback: this never overwrites operator-derived arcs.
            if let Some(args) = function_arguments(name, "travel-cost") {
                if args.len() == 2 && value.is_finite() && value >= 0.0 {
                    fallback_travel.push((args[0].clone(), args[1].clone(), value));
                }
            }
        }

        let load_id = load_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not discover numeric variable load()"
                .to_string()
                .into()
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not discover numeric variable vehicles()"
                .to_string()
                .into()
        })?;
        let capacity = capacity.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not recover the static CVRP capacity"
                .to_string()
                .into()
        })?;
        let max_vehicles = max_vehicles.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not recover the static CVRP max_vehicles value"
                .to_string()
                .into()
        })?;

        if capacity <= 0.0 || max_vehicles < 1.0 {
            return Err(
                "llm_001 found invalid CVRP capacity or fleet limit"
                    .to_string()
                    .into(),
            );
        }

        let mut customers = Vec::<CustomerInfo>::with_capacity(customer_names.len());
        for name in customer_names.iter() {
            let visited_fact = visited_facts
                .get(name)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("llm_001 could not find semantic visited fact for {name}").into()
                })?;
            let loc_fact = location_facts
                .get(name)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("llm_001 could not find semantic loc fact for {name}").into()
                })?;
            let demand = demands
                .get(name)
                .copied()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("llm_001 could not recover demand for customer {name}").into()
                })?;
            if demand > capacity + 1.0e-9 {
                return Err(
                    format!("llm_001 found a customer demand larger than capacity: {name}")
                        .into(),
                );
            }
            customers.push(CustomerInfo {
                visited_fact,
                loc_fact,
                demand,
            });
        }

        let place_index = |name: &str| -> Option<usize> {
            if name == depot_name {
                Some(depot_index)
            } else {
                customer_index.get(name).copied()
            }
        };

        let mut direct_cost = vec![vec![INF; place_count]; place_count];
        let mut via_depot_cost = vec![vec![INF; place_count]; place_count];
        let mut return_to_depot = vec![INF; customer_names.len()];
        for i in 0..place_count {
            direct_cost[i][i] = 0.0;
        }

        // Phase A: primary extraction from grounded operators.
        for (from, to, cost) in visit_records {
            if let (Some(i), Some(j)) = (place_index(&from), place_index(&to)) {
                direct_cost[i][j] = direct_cost[i][j].min(cost);
            }
        }
        for (from, to, cost) in via_records {
            if let (Some(i), Some(j)) = (place_index(&from), place_index(&to)) {
                via_depot_cost[i][j] = via_depot_cost[i][j].min(cost);
            }
        }
        for (from, cost) in return_records {
            if let Some(i) = customer_index.get(&from).copied() {
                return_to_depot[i] = return_to_depot[i].min(cost);
                direct_cost[i][depot_index] = direct_cost[i][depot_index].min(cost);
            }
        }

        // Phase B: fill only missing directed travel arcs from preserved PNEs.
        for (from, to, cost) in fallback_travel {
            if let (Some(i), Some(j)) = (place_index(&from), place_index(&to)) {
                if !direct_cost[i][j].is_finite() {
                    direct_cost[i][j] = cost;
                }
                if j == depot_index && i < customer_names.len() && !return_to_depot[i].is_finite() {
                    return_to_depot[i] = cost;
                }
            }
        }

        // Phase C: reconstruct missing via-depot action costs from directed
        // customer->depot and depot->customer legs. No symmetry is assumed.
        for from in 0..customer_names.len() {
            for to in 0..customer_names.len() {
                if from != to && !via_depot_cost[from][to].is_finite() {
                    let back = return_to_depot[from];
                    let out = direct_cost[depot_index][to];
                    if back.is_finite() && out.is_finite() {
                        via_depot_cost[from][to] = back + out;
                    }
                }
            }
        }

        let usable_depot_arcs = (0..customer_names.len()).all(|i| {
            direct_cost[depot_index][i].is_finite() && return_to_depot[i].is_finite()
        });
        if !usable_depot_arcs {
            return Err(
                "llm_001 could not construct usable directed depot travel arcs"
                    .to_string()
                    .into(),
            );
        }

        let mut min_incoming = vec![INF; customer_names.len()];
        for to in 0..customer_names.len() {
            for from in 0..place_count {
                if from != to {
                    min_incoming[to] = min_incoming[to].min(direct_cost[from][to]);
                }
            }
            if !min_incoming[to].is_finite() {
                return Err(
                    format!("llm_001 has no usable incoming arc for customer index {to}")
                        .into(),
                );
            }
        }

        let mut max_arc: f64 = 1.0;
        for row in direct_cost.iter() {
            for cost in row.iter().copied().filter(|c| c.is_finite()) {
                max_arc = max_arc.max(cost);
            }
        }
        for row in via_depot_cost.iter() {
            for cost in row.iter().copied().filter(|c| c.is_finite()) {
                max_arc = max_arc.max(cost);
            }
        }
        let dead_end_penalty =
            (max_arc * (customer_names.len() as f64 + 1.0) * 1_000_000.0)
                .min(1.0e15)
                .max(1.0e9);

        Ok(Self {
            customers,
            depot_loc_fact,
            depot_index,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            direct_cost,
            via_depot_cost,
            return_to_depot,
            min_incoming,
            dead_end_penalty,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_remaining: RefCell::new(vec![false; customer_names.len()]),
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
        let view = registry.view(state);

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        registry.view(state).fill_numeric(&mut numeric_values);
        let load = numeric_values[self.load_id].max(0.0);
        let vehicles_used = numeric_values[self.vehicles_id].max(0.0);
        drop(numeric_values);

        let at_depot = self.depot_loc_fact.is_hold(view);
        let mut current = if at_depot { Some(self.depot_index) } else { None };

        let mut remaining = self.scratch_remaining.borrow_mut();
        remaining.fill(false);

        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        for (i, customer) in self.customers.iter().enumerate() {
            if customer.loc_fact.is_hold(view) {
                current = Some(i);
            }
            if !customer.visited_fact.is_hold(view) {
                remaining[i] = true;
                remaining_count += 1;
                remaining_demand += customer.demand;
            }
        }

        let current = match current {
            Some(index) => index,
            None => return Ok(self.dead_end_penalty),
        };

        // Every valid completed plan must explicitly return to the depot.
        if remaining_count == 0 {
            if current == self.depot_index {
                return Ok(0.0);
            }
            let back = self.return_to_depot[current];
            return Ok(if back.is_finite() {
                back.max(0.0)
            } else {
                self.dead_end_penalty
            });
        }

        let residual_capacity = (self.capacity - load).max(0.0);
        let remaining_vehicles = (self.max_vehicles - vehicles_used).max(0.0);
        let total_available_capacity = residual_capacity + remaining_vehicles * self.capacity;

        // Resource dead-end detection using the current route's residual load
        // plus every vehicle that may still be opened by visit-via-depot.
        if remaining_demand > total_available_capacity + 1.0e-9 {
            return Ok(self.dead_end_penalty);
        }

        // A demand-only route count is cheap and reacts to heterogeneous
        // demands, residual load, and fleet pressure.
        let demand_after_current_route = (remaining_demand - residual_capacity).max(0.0);
        let mandatory_reloads = ceil_nonnegative(demand_after_current_route / self.capacity);
        if mandatory_reloads > remaining_vehicles + 1.0e-9 {
            return Ok(self.dead_end_penalty);
        }

        // Component 1: directed geometric routing estimate. Re-anchor the
        // estimate at the actual current location and choose the cheapest legal
        // first domain transition. Direct and via-depot transitions are
        // evaluated separately under their respective resource conditions.
        let mut first_customer = None;
        let mut first_cost = INF;
        let mut min_final_return = INF;
        let mut continuation = 0.0;
        let mut min_route_activation = INF;

        for i in 0..self.customers.len() {
            if !remaining[i] {
                continue;
            }

            continuation += self.min_incoming[i];
            min_final_return = min_final_return.min(self.return_to_depot[i]);
            min_route_activation = min_route_activation.min(
                self.direct_cost[self.depot_index][i] + self.return_to_depot[i],
            );

            let demand = self.customers[i].demand;
            let mut transition = INF;

            // The ordinary visit action is available only when load permits.
            if load + demand <= self.capacity + 1.0e-9 {
                transition = transition.min(self.direct_cost[current][i]);
            }

            // visit-via-depot is available only from a customer and only if a
            // new vehicle remains. Its extracted grounded action cost is used.
            if current != self.depot_index && remaining_vehicles >= 1.0 - 1.0e-9 {
                transition = transition.min(self.via_depot_cost[current][i]);
            }

            if transition < first_cost {
                first_cost = transition;
                first_customer = Some(i);
            }
        }

        let first_customer = match first_customer {
            Some(index) if first_cost.is_finite() => index,
            _ => return Ok(self.dead_end_penalty),
        };

        continuation = (continuation - self.min_incoming[first_customer]).max(0.0);
        if !min_final_return.is_finite() {
            return Ok(self.dead_end_penalty);
        }

        let h_routing = first_cost + continuation + min_final_return;

        // Component 2: capacity-route activation estimate. Each route that
        // cannot fit in the current residual capacity incurs a fraction of the
        // cheapest remaining depot-out-and-back connection. This intentionally
        // need not be admissible; it distinguishes states that have consumed
        // different amounts of load and fleet.
        let activation_scale = if min_route_activation.is_finite() {
            0.5 * min_route_activation
        } else {
            return Ok(self.dead_end_penalty);
        };
        let h_capacity = mandatory_reloads * activation_scale;

        // Component 3: smooth fleet-pressure term. States requiring most of
        // their remaining vehicles are less attractive than states with spare
        // route capacity, even when their geometric customer sets are similar.
        let h_fleet = if mandatory_reloads > 0.0 {
            let denominator = remaining_vehicles.max(1.0);
            0.1 * activation_scale * (mandatory_reloads / denominator)
        } else {
            0.0
        };

        drop(remaining);

        let h = h_routing + h_capacity + h_fleet;
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
        },
    ])
}
