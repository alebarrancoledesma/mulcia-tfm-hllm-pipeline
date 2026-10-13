use std::cell::RefCell;
use std::collections::HashMap;

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


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const INF: f64 = f64::INFINITY;
const DEAD_END_PENALTY: f64 = 1.0e15;

#[derive(Clone)]
struct CustomerInfo {
    loc_fact: ExplicitFact,
    visited_fact: ExplicitFact,
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

    // Directed costs. The final index is the depot; customer indices agree
    // with `customers`.
    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    min_outgoing: Vec<f64>,
    min_route_break_cost: f64,
    cost_scale: f64,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

fn parse_atom_argument(name: &str, predicate: &str) -> Option<String> {
    let needle = format!("{}(", predicate);
    let start = name.find(&needle)? + needle.len();
    let rest = &name[start..];
    let end = rest.find(')')?;
    let argument = rest[..end].trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn parse_function_argument(name: &str, function: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let needle = format!("{}(", function);
    let start = lower.find(&needle)? + needle.len();
    let rest = &name[start..];
    let end = rest.find(')')?;
    let argument = rest[..end].trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn parse_binary_function(name: &str, function: &str) -> Option<(String, String)> {
    let lower = name.to_ascii_lowercase();
    let needle = format!("{}(", function);
    let start = lower.find(&needle)? + needle.len();
    let rest = &name[start..];
    let end = rest.find(')')?;
    let mut arguments = rest[..end].split(',').map(str::trim);
    let first = arguments.next()?;
    let second = arguments.next()?;
    if first.is_empty() || second.is_empty() || arguments.next().is_some() {
        None
    } else {
        Some((first.to_string(), second.to_string()))
    }
}

fn compact_lower(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn clean_operator_token(token: &str) -> &str {
    token.trim_matches(|c| c == '(' || c == ')' || c == ',')
}

fn set_min(matrix: &mut [Vec<f64>], from: usize, to: usize, value: f64) {
    if value.is_finite() && value >= 0.0 && value < matrix[from][to] {
        matrix[from][to] = value;
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover semantic facts rather than relying on SAS variable/value IDs.
        let mut loc_facts: HashMap<String, ExplicitFact> = HashMap::new();
        let mut visited_facts: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let variable = &task.variables()[var_id];
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(place) = parse_atom_argument(fact_name, "loc") {
                    loc_facts.insert(place, fact.clone());
                }
                if let Some(customer) = parse_atom_argument(fact_name, "visited") {
                    visited_facts.insert(customer, fact.clone());
                }
            }
        }

        if visited_facts.is_empty() {
            return Err("llm_001 could not discover any visited(customer) facts"
                .to_string()
                .into());
        }

        // A depot has a loc fact but no corresponding visited fact.
        let mut depot_names: Vec<String> = loc_facts
            .keys()
            .filter(|name| !visited_facts.contains_key(*name))
            .cloned()
            .collect();
        depot_names.sort();
        if depot_names.len() != 1 {
            return Err(format!(
                "llm_001 expected one depot-like loc fact, found {}",
                depot_names.len()
            )
            .into());
        }
        let depot_name = depot_names.remove(0);
        let depot_loc_fact = loc_facts
            .get(&depot_name)
            .cloned()
            .ok_or_else(|| "llm_001 lost the discovered depot fact".to_string())?;

        let mut customer_names: Vec<String> = visited_facts.keys().cloned().collect();
        customer_names.sort();
        for name in &customer_names {
            if !loc_facts.contains_key(name) {
                return Err(format!(
                    "llm_001 found visited({name}) but no matching loc({name}) fact"
                )
                .into());
            }
        }

        // Resolve all relevant numeric IDs and static resource values once.
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut max_vehicles: Option<f64> = None;
        let mut demands: HashMap<String, f64> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            let compact = compact_lower(name);

            if compact.ends_with("load()") && !compact.contains("derived!") {
                load_id = Some(id);
            }
            if compact.ends_with("vehicles()")
                && !compact.ends_with("max_vehicles()")
                && !compact.contains("derived!")
            {
                vehicles_id = Some(id);
            }
            if compact.ends_with("capacity()")
                && !compact.ends_with("max_capacity()")
                && id < initial_numeric.len()
            {
                capacity = Some(initial_numeric[id]);
            }
            if compact.ends_with("max_vehicles()") && id < initial_numeric.len() {
                max_vehicles = Some(initial_numeric[id]);
            }
            if let Some(customer) = parse_function_argument(name, "demand") {
                if id < initial_numeric.len() {
                    demands.insert(customer, initial_numeric[id]);
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            HeuristicBuildError::from("llm_001 could not resolve the dynamic load() numeric variable".to_string())
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            HeuristicBuildError::from("llm_001 could not resolve the dynamic vehicles() numeric variable".to_string())
        })?;
        let capacity = capacity.ok_or_else(|| {
            HeuristicBuildError::from("llm_001 could not recover capacity() from numeric task data".to_string())
        })?;
        let max_vehicles = max_vehicles.ok_or_else(|| {
            HeuristicBuildError::from("llm_001 could not recover max_vehicles() from numeric task data".to_string())
        })?;

        if !capacity.is_finite() || capacity <= 0.0 {
            return Err("llm_001 recovered a non-positive or non-finite capacity"
                .to_string()
                .into());
        }
        if !max_vehicles.is_finite() || max_vehicles < 1.0 {
            return Err("llm_001 recovered an invalid max_vehicles value"
                .to_string()
                .into());
        }

        let mut customers = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let demand = demands.get(name).copied().ok_or_else(|| {
                HeuristicBuildError::from(format!(
                    "llm_001 could not recover demand for discovered customer {name}"
                ))
            })?;
            if !demand.is_finite() || demand < 0.0 {
                return Err(format!("llm_001 recovered an invalid demand for {name}").into());
            }
            customers.push(CustomerInfo {
                loc_fact: loc_facts[name].clone(),
                visited_fact: visited_facts[name].clone(),
                demand,
            });
        }

        let customer_count = customers.len();
        let depot_index = customer_count;
        let node_count = customer_count + 1;
        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }
        node_index.insert(depot_name.clone(), depot_index);

        let mut direct_cost = vec![vec![INF; node_count]; node_count];
        let mut via_depot_cost = vec![vec![INF; node_count]; node_count];
        for i in 0..node_count {
            direct_cost[i][i] = 0.0;
        }

        // Phase A: primary extraction from grounded routing operators.
        for operator in task.get_operators() {
            let tokens: Vec<&str> = operator
                .name()
                .split_whitespace()
                .map(clean_operator_token)
                .filter(|token| !token.is_empty())
                .collect();
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
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        set_min(&mut direct_cost, from, to, cost);
                    }
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        set_min(&mut via_depot_cost, from, to, cost);
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = node_index.get(tokens[1]) {
                        set_min(&mut direct_cost, from, depot_index, cost);
                    }
                }
                _ => {}
            }
        }

        // Phase B: optional PNE fallback for only still-missing direct arcs.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            if id >= initial_numeric.len() {
                continue;
            }
            if let Some((from_name, to_name)) =
                parse_binary_function(variable.name(), "travel-cost")
            {
                if let (Some(&from), Some(&to)) =
                    (node_index.get(&from_name), node_index.get(&to_name))
                {
                    if !direct_cost[from][to].is_finite() {
                        set_min(&mut direct_cost, from, to, initial_numeric[id]);
                    }
                }
            }
        }

        // Every customer must at least be reachable from, and returnable to, the depot.
        for customer in 0..customer_count {
            if !direct_cost[depot_index][customer].is_finite()
                || !direct_cost[customer][depot_index].is_finite()
            {
                return Err(format!(
                    "llm_001 routing matrix is unusable: missing depot arc for customer index {customer}"
                )
                .into());
            }
        }

        // Phase C: fill only missing customer-customer arcs via the depot.
        for from in 0..customer_count {
            for to in 0..customer_count {
                if from == to {
                    continue;
                }
                let depot_composite =
                    direct_cost[from][depot_index] + direct_cost[depot_index][to];
                if !via_depot_cost[from][to].is_finite() && depot_composite.is_finite() {
                    via_depot_cost[from][to] = depot_composite;
                }
                if !direct_cost[from][to].is_finite() && depot_composite.is_finite() {
                    // Conservative connectivity fallback. Capacity simulation still
                    // distinguishes this from an actual via-depot transition.
                    direct_cost[from][to] = depot_composite;
                }
            }
        }

        let mut min_outgoing = vec![INF; customer_count];
        let mut finite_cost_sum = 0.0;
        let mut finite_cost_count = 0usize;
        for from in 0..customer_count {
            for to in 0..customer_count {
                if from != to && direct_cost[from][to].is_finite() {
                    min_outgoing[from] = min_outgoing[from].min(direct_cost[from][to]);
                }
            }
            min_outgoing[from] = min_outgoing[from]
                .min(direct_cost[from][depot_index]);
            if !min_outgoing[from].is_finite() {
                return Err(format!(
                    "llm_001 found no usable outgoing cost for customer index {from}"
                )
                .into());
            }

            if direct_cost[depot_index][from].is_finite() {
                finite_cost_sum += direct_cost[depot_index][from];
                finite_cost_count += 1;
            }
            if direct_cost[from][depot_index].is_finite() {
                finite_cost_sum += direct_cost[from][depot_index];
                finite_cost_count += 1;
            }
        }

        let cost_scale = if finite_cost_count > 0 {
            finite_cost_sum / finite_cost_count as f64
        } else {
            1.0
        };

        let mut min_route_break_cost = INF;
        for from in 0..customer_count {
            for to in 0..customer_count {
                if from != to && via_depot_cost[from][to].is_finite() {
                    min_route_break_cost =
                        min_route_break_cost.min(via_depot_cost[from][to]);
                }
            }
        }
        if !min_route_break_cost.is_finite() {
            min_route_break_cost = cost_scale;
        }

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
            min_outgoing,
            min_route_break_cost,
            cost_scale,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
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
        let state_view = registry.view(state);

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        state_view.fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        if self.load_id >= numeric_values.len() || self.vehicles_id >= numeric_values.len() {
            return Err(EvaluationError::ComputationFailed(
                "llm_001 numeric scratch state is smaller than resolved IDs".to_string(),
            ));
        }

        let load = numeric_values[self.load_id];
        let vehicles_used = numeric_values[self.vehicles_id];
        if !load.is_finite() || !vehicles_used.is_finite() {
            return Ok(DEAD_END_PENALTY);
        }

        let mut current = None;
        if self.depot_loc_fact.is_hold(state_view) {
            current = Some(self.depot_index);
        }

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        if unvisited.len() != self.customers.len() {
            unvisited.resize(self.customers.len(), false);
        }

        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        let mut min_final_return = INF;
        let mut backbone = 0.0;

        for (index, customer) in self.customers.iter().enumerate() {
            if current.is_none() && customer.loc_fact.is_hold(state_view) {
                current = Some(index);
            }

            let is_unvisited = !customer.visited_fact.is_hold(state_view);
            unvisited[index] = is_unvisited;
            if is_unvisited {
                remaining_count += 1;
                remaining_demand += customer.demand;
                backbone += self.min_outgoing[index];
                min_final_return = min_final_return
                    .min(self.direct_cost[index][self.depot_index]);
            }
        }

        let current = match current {
            Some(index) => index,
            None => return Ok(DEAD_END_PENALTY),
        };

        // All customers are served: only the mandatory final return remains.
        if remaining_count == 0 {
            if current == self.depot_index {
                return Ok(0.0);
            }
            let return_cost = self.direct_cost[current][self.depot_index];
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                DEAD_END_PENALTY
            });
        }

        let residual_capacity = (self.capacity - load).max(0.0);
        let remaining_vehicle_slots = (self.max_vehicles - vehicles_used).max(0.0);
        let total_available_capacity =
            residual_capacity + remaining_vehicle_slots * self.capacity;

        // Resource dead-end detection uses the current partially loaded vehicle
        // plus every vehicle that may still be started by visit-via-depot.
        if remaining_demand > total_available_capacity + 1.0e-9 {
            return Ok(DEAD_END_PENALTY);
        }
        for (index, customer) in self.customers.iter().enumerate() {
            if unvisited[index] && customer.demand > self.capacity + 1.0e-9 {
                return Ok(DEAD_END_PENALTY);
            }
        }

        let demand_after_current_route =
            (remaining_demand - residual_capacity).max(0.0);
        let required_new_routes = if demand_after_current_route <= 1.0e-9 {
            0.0
        } else {
            (demand_after_current_route / self.capacity - 1.0e-12).ceil()
        };

        if required_new_routes > remaining_vehicle_slots + 1.0e-9 {
            return Ok(DEAD_END_PENALTY);
        }

        // State-anchored first transition: direct visit is load-constrained;
        // visit-via-depot resets load and consumes another vehicle.
        let mut first_transition = INF;
        for (to, customer) in self.customers.iter().enumerate() {
            if !unvisited[to] {
                continue;
            }

            if load + customer.demand <= self.capacity + 1.0e-9 {
                first_transition =
                    first_transition.min(self.direct_cost[current][to]);
            }

            if current != self.depot_index
                && vehicles_used + 1.0 <= self.max_vehicles + 1.0e-9
            {
                first_transition =
                    first_transition.min(self.via_depot_cost[current][to]);
            }
        }

        if !first_transition.is_finite() || !min_final_return.is_finite() {
            return Ok(DEAD_END_PENALTY);
        }

        // Component 1: directed geometric routing estimate. It is anchored at
        // the current location and explicitly includes the eventual depot return.
        let h_routing = first_transition + backbone + min_final_return;

        // Component 2: capacity-induced route-break estimate. Heterogeneous
        // demands, current load, and capacity determine the number of reloads.
        let h_capacity = required_new_routes * self.min_route_break_cost;

        // Component 3: fleet pressure. This deliberately biases greedy search
        // away from states that leave little fleet slack for substantial demand.
        let slot_denominator = remaining_vehicle_slots + 1.0;
        let demand_routes = remaining_demand / self.capacity;
        let used_fraction =
            (vehicles_used / self.max_vehicles.max(1.0)).clamp(0.0, 1.0);
        let h_vehicles = self.cost_scale
            * (required_new_routes * required_new_routes / slot_denominator
                + 0.05 * demand_routes * used_fraction);

        let h = h_routing + h_capacity + h_vehicles;
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
