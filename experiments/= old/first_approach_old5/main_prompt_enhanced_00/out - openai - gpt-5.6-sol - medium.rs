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

const MISSING_COST: f64 = f64::MAX;

struct CustomerInfo {
    visited_var: usize,
    visited_value: usize,
    demand: f64,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    remaining: Vec<bool>,
}

struct LlmHeuristic {
    // Node zero is the depot; nodes 1.. are customers.
    customers: Vec<CustomerInfo>,
    location_facts: Vec<Option<(usize, usize)>>,
    direct_cost: Vec<f64>,
    via_depot_cost: Vec<f64>,
    node_count: usize,
    load_var: usize,
    capacity: f64,
    scratch: RefCell<ScratchBuffers>,
}

fn parse_atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let trimmed = name.trim();
    let prefix = format!("Atom {}(", predicate);
    if !trimmed.starts_with(&prefix) || !trimmed.ends_with(')') {
        return None;
    }

    let argument = &trimmed[prefix.len()..trimmed.len() - 1];
    let argument = argument.trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument)
    }
}

fn exact_numeric_name(name: &str, bare_name: &str) -> bool {
    let trimmed = name.trim();
    trimmed == bare_name || trimmed == format!("PNE {}", bare_name)
}

fn usable_cost(cost: f64) -> Option<f64> {
    if cost.is_finite() && cost >= 0.0 {
        Some(cost)
    } else {
        None
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut visited_by_name: HashMap<String, (usize, usize)> = HashMap::new();
        let mut location_by_name: HashMap<String, (usize, usize)> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = parse_atom_argument(fact_name, "visited") {
                    visited_by_name
                        .entry(customer.to_string())
                        .or_insert((var_id, value));
                }

                if let Some(place) = parse_atom_argument(fact_name, "loc") {
                    location_by_name
                        .entry(place.to_string())
                        .or_insert((var_id, value));
                }
            }
        }

        if visited_by_name.is_empty() {
            return Err(
                "llm_001 could not discover any CVRP visited-customer facts"
                    .to_string()
                    .into(),
            );
        }

        let mut depot_name: Option<String> = None;
        for goal_id in 0..task.get_num_goals() {
            let goal_fact = task.get_goal_fact(goal_id);
            let goal_name = task.get_fact_name(&goal_fact);
            if let Some(place) = parse_atom_argument(goal_name, "loc") {
                if !visited_by_name.contains_key(place) {
                    depot_name = Some(place.to_string());
                    break;
                }
            }
        }

        if depot_name.is_none() {
            for place in location_by_name.keys() {
                if !visited_by_name.contains_key(place) {
                    depot_name = Some(place.clone());
                    break;
                }
            }
        }

        let depot_name = match depot_name {
            Some(name) => name,
            None => {
                return Err(
                    "llm_001 could not identify the CVRP depot location"
                        .to_string()
                        .into(),
                );
            }
        };

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());

        let node_count = node_names.len();
        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let mut location_facts = vec![None; node_count];
        for (index, name) in node_names.iter().enumerate() {
            if let Some(&(var_id, value)) = location_by_name.get(name) {
                location_facts[index] = Some((var_id, value));
            }
        }

        if location_facts[0].is_none() {
            return Err(
                "llm_001 could not discover the depot location fact"
                    .to_string()
                    .into(),
            );
        }

        let mut load_var: Option<usize> = None;
        let mut capacity_var: Option<usize> = None;
        let mut demand_vars: HashMap<String, usize> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name().trim();

            if exact_numeric_name(name, "load()") {
                load_var = Some(id);
            } else if exact_numeric_name(name, "capacity()") {
                capacity_var = Some(id);
            } else {
                for customer in &customer_names {
                    let demand_name = format!("demand({})", customer);
                    if exact_numeric_name(name, &demand_name) {
                        demand_vars.insert(customer.clone(), id);
                        break;
                    }
                }
            }
        }

        let load_var = match load_var {
            Some(id) => id,
            None => {
                return Err(
                    "llm_001 could not find the required exact load() numeric fluent"
                        .to_string()
                        .into(),
                );
            }
        };

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_var
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(f64::MAX);

        let mut customers = Vec::with_capacity(customer_names.len());
        for customer in &customer_names {
            let (visited_var, visited_value) = match visited_by_name.get(customer) {
                Some(&(var_id, value)) => (var_id, value),
                None => {
                    return Err(
                        "llm_001 lost a discovered customer fact during construction"
                            .to_string()
                            .into(),
                    );
                }
            };

            let demand = demand_vars
                .get(customer)
                .and_then(|id| initial_numeric.get(*id).copied())
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(0.0);

            customers.push(CustomerInfo {
                visited_var,
                visited_value,
                demand,
            });
        }

        let matrix_size = node_count.saturating_mul(node_count);
        let mut direct_cost = vec![MISSING_COST; matrix_size];
        let mut via_depot_cost = vec![MISSING_COST; matrix_size];

        for i in 0..node_count {
            direct_cost[i * node_count + i] = 0.0;
            via_depot_cost[i * node_count + i] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens: Vec<&str> = operator.name().split_whitespace().collect();
            let metric_cost = match usable_cost(
                metric_operator_cost_from_initial_values(task, operator),
            ) {
                Some(cost) => cost,
                None => continue,
            };

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_index.get(*from), node_index.get(*to))
                    {
                        let slot = from_id * node_count + to_id;
                        if metric_cost < direct_cost[slot] {
                            direct_cost[slot] = metric_cost;
                        }
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_index.get(*from), node_index.get(*to))
                    {
                        let slot = from_id * node_count + to_id;
                        if metric_cost < via_depot_cost[slot] {
                            via_depot_cost[slot] = metric_cost;
                        }
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = node_index.get(*from) {
                        let slot = from_id * node_count;
                        if metric_cost < direct_cost[slot] {
                            direct_cost[slot] = metric_cost;
                        }
                    }
                }
                _ => {}
            }
        }

        // If a via-depot operator cost is unavailable, reconstruct it from the
        // two directed legs when both were recovered from grounded operators.
        for from in 1..node_count {
            for to in 1..node_count {
                if from == to {
                    continue;
                }
                let slot = from * node_count + to;
                if via_depot_cost[slot] == MISSING_COST {
                    let first = direct_cost[from * node_count];
                    let second = direct_cost[to];
                    if first != MISSING_COST && second != MISSING_COST {
                        let sum = first + second;
                        if sum.is_finite() && sum >= 0.0 {
                            via_depot_cost[slot] = sum;
                        }
                    }
                }
            }
        }

        Ok(Self {
            customers,
            location_facts,
            direct_cost,
            via_depot_cost,
            node_count,
            load_var,
            capacity,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                numeric: vec![0.0; task.numeric_variables().len()],
                remaining: vec![false; customer_names.len()],
            }),
        })
    }

    fn cost(&self, matrix: &[f64], from: usize, to: usize) -> f64 {
        let index = from.saturating_mul(self.node_count).saturating_add(to);
        matrix.get(index).copied().unwrap_or(MISSING_COST)
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
        let mut scratch = self.scratch.borrow_mut();

        state.fill_state(registry, &mut scratch.propositional);
        registry.view(state).fill_numeric(&mut scratch.numeric).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let mut remaining_count = 0usize;
        for customer_id in 0..self.customers.len() {
            let customer = &self.customers[customer_id];
            let is_visited = scratch
                .propositional
                .get(customer.visited_var)
                .map(|value| *value == customer.visited_value)
                .unwrap_or(false);
            scratch.remaining[customer_id] = !is_visited;
            if !is_visited {
                remaining_count += 1;
            }
        }

        let mut current_node = 0usize;
        for node in 0..self.location_facts.len() {
            if let Some((var_id, value)) = self.location_facts[node] {
                let holds = scratch
                    .propositional
                    .get(var_id)
                    .map(|current| *current == value)
                    .unwrap_or(false);
                if holds {
                    current_node = node;
                    break;
                }
            }
        }

        let mut simulated_load = scratch
            .numeric
            .get(self.load_var)
            .copied()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.0);

        if remaining_count == 0 {
            let return_cost = self.cost(&self.direct_cost, current_node, 0);
            if return_cost != MISSING_COST && return_cost.is_finite() {
                return Ok(return_cost.max(0.0));
            }
            return Ok(0.0);
        }

        let mut estimate = 0.0f64;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = MISSING_COST;
            let mut best_uses_depot = false;

            for customer_id in 0..self.customers.len() {
                if !scratch.remaining[customer_id] {
                    continue;
                }

                let to_node = customer_id + 1;
                let demand = self.customers[customer_id].demand;
                let direct = self.cost(&self.direct_cost, current_node, to_node);
                let direct_feasible = self.capacity == f64::MAX
                    || simulated_load + demand <= self.capacity + 1.0e-9;

                if direct_feasible && direct < best_cost {
                    best_cost = direct;
                    best_customer = Some(customer_id);
                    best_uses_depot = false;
                }

                if current_node != 0 {
                    let via = self.cost(&self.via_depot_cost, current_node, to_node);
                    if via < best_cost {
                        best_cost = via;
                        best_customer = Some(customer_id);
                        best_uses_depot = true;
                    }
                }
            }

            let chosen = match best_customer {
                Some(customer_id) => customer_id,
                None => {
                    // Missing static information should weaken the estimate,
                    // not cause a panic or incorrectly mark a dead end.
                    break;
                }
            };

            if best_cost == MISSING_COST || !best_cost.is_finite() || best_cost < 0.0 {
                break;
            }

            if estimate > f64::MAX / 4.0 - best_cost {
                estimate = f64::MAX / 4.0;
                break;
            }
            estimate += best_cost;

            let demand = self.customers[chosen].demand;
            if best_uses_depot {
                simulated_load = demand;
            } else {
                simulated_load += demand;
            }

            current_node = chosen + 1;
            scratch.remaining[chosen] = false;
            remaining_count -= 1;
        }

        let return_cost = self.cost(&self.direct_cost, current_node, 0);
        if return_cost != MISSING_COST && return_cost.is_finite() && return_cost >= 0.0 {
            if estimate <= f64::MAX / 4.0 - return_cost {
                estimate += return_cost;
            } else {
                estimate = f64::MAX / 4.0;
            }
        }

        if estimate.is_finite() && estimate >= 0.0 {
            Ok(estimate)
        } else {
            Ok(0.0)
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
