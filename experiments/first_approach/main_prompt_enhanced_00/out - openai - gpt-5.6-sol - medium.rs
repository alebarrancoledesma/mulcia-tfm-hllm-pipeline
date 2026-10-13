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
use std::collections::{BTreeSet, HashMap};


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const MAX_FINITE_HEURISTIC: f64 = 1.0e300_f64;

struct CustomerInfo {
    visited_var: usize,
    visited_value: usize,
    loc_var: usize,
    loc_value: usize,
    demand: f64,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    remaining: Vec<bool>,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    depot_loc_var: usize,
    depot_loc_value: usize,
    distance: Vec<f64>,
    node_count: usize,
    depot_index: usize,
    load_id: Option<usize>,
    capacity_id: Option<usize>,
    scratch: RefCell<ScratchBuffers>,
}

fn parse_atom_argument(name: &str, predicate: &str) -> Option<String> {
    let trimmed = name.trim();
    let atom = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);
    let argument = atom.strip_prefix(&prefix)?.strip_suffix(')')?;
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn canonical_numeric_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed)
}

fn parse_numeric_argument(name: &str, function: &str) -> Option<String> {
    let canonical = canonical_numeric_name(name);
    let prefix = format!("{}(", function);
    let argument = canonical.strip_prefix(&prefix)?.strip_suffix(')')?;
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

fn safe_cost(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0_f64 {
        Some(value.min(MAX_FINITE_HEURISTIC))
    } else {
        None
    }
}

fn saturated_add(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return MAX_FINITE_HEURISTIC;
    }
    let result = a + b;
    if result.is_finite() {
        result.min(MAX_FINITE_HEURISTIC)
    } else {
        MAX_FINITE_HEURISTIC
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut visited_facts: HashMap<String, (usize, usize)> = HashMap::new();
        let mut loc_facts: HashMap<String, (usize, usize)> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(object) = parse_atom_argument(&fact_name, "visited") {
                    visited_facts.insert(object, (var_id, value));
                }
                if let Some(object) = parse_atom_argument(&fact_name, "loc") {
                    loc_facts.insert(object, (var_id, value));
                }
            }
        }

        if visited_facts.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not discover any visited(customer) facts".to_string(),
            ));
        }

        let customer_names: BTreeSet<String> = visited_facts.keys().cloned().collect();
        let depot_candidates: Vec<String> = loc_facts
            .keys()
            .filter(|name| !customer_names.contains(*name))
            .cloned()
            .collect();

        if depot_candidates.len() != 1 {
            return Err(HeuristicBuildError::from(format!(
                "llm_001 expected exactly one depot location fact, found {}",
                depot_candidates.len()
            )));
        }

        let depot_name = depot_candidates[0].clone();
        let (depot_loc_var, depot_loc_value) = match loc_facts.get(&depot_name) {
            Some(pair) => *pair,
            None => {
                return Err(HeuristicBuildError::from(
                    "llm_001 lost the discovered depot location fact".to_string(),
                ));
            }
        };

        let ordered_names: Vec<String> = customer_names.into_iter().collect();
        let mut customer_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in ordered_names.iter().enumerate() {
            customer_index.insert(name.clone(), index);
        }

        let mut load_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut exact_demands: HashMap<String, f64> = HashMap::new();
        let initial_numeric = task.get_initial_numeric_state_values();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let canonical = canonical_numeric_name(variable.name());
            if canonical == "load()" {
                load_id = Some(id);
            } else if canonical == "capacity()" {
                capacity_id = Some(id);
            } else if let Some(customer) = parse_numeric_argument(variable.name(), "demand") {
                if let Some(value) = initial_numeric.get(id).copied().and_then(safe_cost) {
                    exact_demands.insert(customer, value);
                }
            }
        }

        let mut demands: Vec<Option<f64>> = vec![None; ordered_names.len()];
        for (name, value) in exact_demands {
            if let Some(index) = customer_index.get(&name).copied() {
                demands[index] = Some(value);
            }
        }

        let depot_index = ordered_names.len();
        let node_count = depot_index + 1;
        let mut distance = vec![f64::NAN; node_count * node_count];
        for node in 0..node_count {
            distance[node * node_count + node] = 0.0_f64;
        }

        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            let metric_cost = safe_cost(metric_operator_cost_from_initial_values(task, operator));

            if tokens.len() == 3 && tokens[0] == "visit" {
                let from = if tokens[1] == depot_name {
                    Some(depot_index)
                } else {
                    customer_index.get(tokens[1]).copied()
                };
                let to = customer_index.get(tokens[2]).copied();

                if let (Some(from_index), Some(to_index), Some(cost)) = (from, to, metric_cost) {
                    let slot = from_index * node_count + to_index;
                    if !distance[slot].is_finite() || cost < distance[slot] {
                        distance[slot] = cost;
                    }
                }
            } else if tokens.len() == 2 && tokens[0] == "return-to-depot" {
                if let (Some(from_index), Some(cost)) =
                    (customer_index.get(tokens[1]).copied(), metric_cost)
                {
                    let slot = from_index * node_count + depot_index;
                    if !distance[slot].is_finite() || cost < distance[slot] {
                        distance[slot] = cost;
                    }
                }
            } else if tokens.len() == 3 && tokens[0] == "visit-via-depot" {
                if let (Some(load_var), Some(to_index)) =
                    (load_id, customer_index.get(tokens[2]).copied())
                {
                    for assignment in operator.assignment_effects() {
                        if assignment.affected_var_id() == load_var {
                            let rhs_id = assignment.var_id();
                            if let Some(value) = initial_numeric
                                .get(rhs_id)
                                .copied()
                                .and_then(safe_cost)
                            {
                                match demands[to_index] {
                                    None => demands[to_index] = Some(value),
                                    Some(previous) => {
                                        if (previous - value).abs() <= 1.0e-9_f64 {
                                            demands[to_index] = Some(previous);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut customers = Vec::with_capacity(ordered_names.len());
        for name in &ordered_names {
            let index = match customer_index.get(name).copied() {
                Some(index) => index,
                None => {
                    return Err(HeuristicBuildError::from(
                        "llm_001 failed to index a discovered customer".to_string(),
                    ));
                }
            };
            let (visited_var, visited_value) = match visited_facts.get(name).copied() {
                Some(pair) => pair,
                None => {
                    return Err(HeuristicBuildError::from(format!(
                        "llm_001 is missing the visited fact for {}",
                        name
                    )));
                }
            };
            let (loc_var, loc_value) = match loc_facts.get(name).copied() {
                Some(pair) => pair,
                None => {
                    return Err(HeuristicBuildError::from(format!(
                        "llm_001 is missing the location fact for {}",
                        name
                    )));
                }
            };

            customers.push(CustomerInfo {
                visited_var,
                visited_value,
                loc_var,
                loc_value,
                demand: demands[index].unwrap_or(0.0_f64),
            });
        }

        let mut missing_edges = 0_usize;
        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !distance[from * node_count + to].is_finite() {
                    missing_edges += 1;
                }
            }
        }
        if missing_edges > 0 {
            return Err(HeuristicBuildError::from(format!(
                "llm_001 could not reconstruct {} directed routing costs from grounded operators",
                missing_edges
            )));
        }

        Ok(Self {
            customers,
            depot_loc_var,
            depot_loc_value,
            distance,
            node_count,
            depot_index,
            load_id,
            capacity_id,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                numeric: vec![0.0_f64; task.numeric_variables().len()],
                remaining: vec![false; ordered_names.len()],
            }),
        })
    }

    fn distance(&self, from: usize, to: usize) -> f64 {
        let value = self.distance[from * self.node_count + to];
        if value.is_finite() && value >= 0.0_f64 {
            value.min(MAX_FINITE_HEURISTIC)
        } else {
            0.0_f64
        }
    }
}


impl Heuristic for LlmHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if eval_state.is_goal() {
            return Ok(0.0_f64);
        }

        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let mut scratch = self.scratch.borrow_mut();

        state.fill_state(registry, &mut scratch.propositional);
        registry
            .view(state)
            .fill_numeric(&mut scratch.numeric)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

        let mut current = self.depot_index;
        if scratch.propositional[self.depot_loc_var] != self.depot_loc_value {
            for (index, customer) in self.customers.iter().enumerate() {
                if scratch.propositional[customer.loc_var] == customer.loc_value {
                    current = index;
                    break;
                }
            }
        }

        let mut remaining_count = 0_usize;
        for (index, customer) in self.customers.iter().enumerate() {
            let unvisited = scratch.propositional[customer.visited_var]
                != customer.visited_value;
            scratch.remaining[index] = unvisited;
            if unvisited {
                remaining_count += 1;
            }
        }

        if remaining_count == 0 {
            return Ok(self.distance(current, self.depot_index));
        }

        let mut load = self
            .load_id
            .and_then(|id| scratch.numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0_f64)
            .unwrap_or(0.0_f64);

        let capacity = self
            .capacity_id
            .and_then(|id| scratch.numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0_f64)
            .unwrap_or(f64::MAX);

        let mut h: f64 = 0.0_f64;
        let mut from = current;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = MAX_FINITE_HEURISTIC;
            let mut best_new_load = load;

            for customer_index in 0..self.customers.len() {
                if !scratch.remaining[customer_index] {
                    continue;
                }

                let demand = self.customers[customer_index].demand;
                let direct_allowed = from == self.depot_index
                    || capacity == f64::MAX
                    || load + demand <= capacity + 1.0e-9_f64;

                let (transition_cost, resulting_load) = if direct_allowed {
                    let new_load = if from == self.depot_index {
                        demand
                    } else {
                        load + demand
                    };
                    (self.distance(from, customer_index), new_load)
                } else {
                    let via_depot = saturated_add(
                        self.distance(from, self.depot_index),
                        self.distance(self.depot_index, customer_index),
                    );
                    (via_depot, demand)
                };

                if best_customer.is_none() || transition_cost < best_cost {
                    best_customer = Some(customer_index);
                    best_cost = transition_cost;
                    best_new_load = resulting_load;
                }
            }

            let chosen = match best_customer {
                Some(index) => index,
                None => break,
            };

            h = saturated_add(h, best_cost);
            load = if best_new_load.is_finite() && best_new_load >= 0.0_f64 {
                best_new_load
            } else {
                0.0_f64
            };
            from = chosen;
            scratch.remaining[chosen] = false;
            remaining_count -= 1;
        }

        h = saturated_add(h, self.distance(from, self.depot_index));

        if h.is_finite() && h >= 0.0_f64 {
            Ok(h.min(MAX_FINITE_HEURISTIC))
        } else {
            Ok(MAX_FINITE_HEURISTIC)
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
