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

use std::collections::HashMap;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

#[derive(Clone)]
struct FactBinding {
    var_id: usize,
    value: usize,
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<FactBinding>,
    location_facts: Vec<FactBinding>,
    depot_location_fact: FactBinding,
    distances: Vec<f64>,
    node_count: usize,
    load_id: Option<usize>,
    capacity: Option<f64>,
    demands: Option<Vec<f64>>,
}

fn parse_atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let trimmed = name.trim();
    let prefix = format!("Atom {}(", predicate);
    let rest = trimmed.strip_prefix(&prefix)?;
    let argument = rest.strip_suffix(')')?;

    if argument.is_empty()
        || argument.contains('(')
        || argument.contains(')')
        || argument.contains(',')
        || argument.chars().any(char::is_whitespace)
    {
        return None;
    }

    Some(argument)
}

fn normalized_pne_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed)
}

fn parse_exact_unary_pne<'a>(name: &'a str, function: &str) -> Option<&'a str> {
    let normalized = normalized_pne_name(name);
    let open = normalized.find('(')?;

    if &normalized[..open] != function || !normalized.ends_with(')') {
        return None;
    }

    let argument = &normalized[(open + 1)..(normalized.len() - 1)];
    if argument.is_empty()
        || argument.contains('(')
        || argument.contains(')')
        || argument.contains(',')
        || argument.chars().any(char::is_whitespace)
    {
        return None;
    }

    Some(argument)
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let initial_propositional = task.get_initial_propositional_state_values();

        let mut visited_by_name: HashMap<String, FactBinding> = HashMap::new();
        let mut location_by_name: HashMap<String, FactBinding> = HashMap::new();
        let mut initially_located_object: Option<String> = None;

        for var_id in 0..task.variables().len() {
            let variable = &task.variables()[var_id];
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = parse_atom_argument(fact_name, "visited") {
                    visited_by_name.insert(
                        object.to_string(),
                        FactBinding { var_id, value },
                    );
                }

                if let Some(object) = parse_atom_argument(fact_name, "loc") {
                    location_by_name.insert(
                        object.to_string(),
                        FactBinding { var_id, value },
                    );

                    if initial_propositional[var_id] == value {
                        initially_located_object = Some(object.to_string());
                    }
                }
            }
        }

        let depot_name = initially_located_object.ok_or_else(|| -> HeuristicBuildError {
            "missing initially true depot location fact".to_string().into()
        })?;

        let depot_location_fact = location_by_name
            .get(&depot_name)
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "missing depot location binding".to_string().into()
            })?;

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP task has no visited customer facts".to_string(),
            ));
        }

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        let mut location_facts = Vec::with_capacity(customer_names.len());

        for customer in &customer_names {
            visited_facts.push(
                visited_by_name
                    .get(customer)
                    .cloned()
                    .ok_or_else(|| -> HeuristicBuildError {
                        format!("missing visited fact for {customer}").into()
                    })?,
            );

            location_facts.push(
                location_by_name
                    .get(customer)
                    .cloned()
                    .ok_or_else(|| -> HeuristicBuildError {
                        format!("missing location fact for {customer}").into()
                    })?,
            );
        }

        // Node zero is the depot; customer i is stored at node i + 1.
        let node_count = customer_names.len() + 1;
        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), 0);
        for (i, customer) in customer_names.iter().enumerate() {
            node_by_name.insert(customer.clone(), i + 1);
        }

        let mut distances = vec![f64::INFINITY; node_count * node_count];
        for i in 0..node_count {
            distances[i * node_count + i] = 0.0;
        }

        // Reconstruct all direct routing costs from grounded operators rather
        // than from potentially folded travel-cost numeric variables.
        for op in task.get_operators() {
            let tokens = operator_tokens(op.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_by_name.get(*from), node_by_name.get(*to))
                    {
                        let index = from_id * node_count + to_id;
                        distances[index] = distances[index].min(cost);
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = node_by_name.get(*from) {
                        let index = from_id * node_count;
                        distances[index] = distances[index].min(cost);
                    }
                }
                _ => {}
            }
        }

        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !distances[from * node_count + to].is_finite() {
                    return Err(HeuristicBuildError::from(format!(
                        "missing direct routing cost for node pair {from} -> {to}"
                    )));
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut demand_by_name: HashMap<String, f64> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let normalized = normalized_pne_name(variable.name());

            if normalized == "load()" {
                load_id = Some(id);
            } else if normalized == "capacity()" {
                let value = initial_numeric[id];
                if value.is_finite() && value > 0.0 {
                    capacity = Some(value);
                }
            } else if let Some(customer) = parse_exact_unary_pne(variable.name(), "demand") {
                let value = initial_numeric[id];
                if value.is_finite() && value >= 0.0 {
                    demand_by_name.insert(customer.to_string(), value);
                }
            }
        }

        // Capacity and demand PNEs may have been folded away by translation.
        // Enable capacity-aware routing only when the complete exact data is
        // available; otherwise retain the routing-only heuristic.
        let demands = if capacity.is_some() && load_id.is_some() {
            let mut values = Vec::with_capacity(customer_names.len());
            let mut complete = true;

            for customer in &customer_names {
                if let Some(value) = demand_by_name.get(customer) {
                    values.push(*value);
                } else {
                    complete = false;
                    break;
                }
            }

            if complete {
                Some(values)
            } else {
                None
            }
        } else {
            None
        };

        let capacity = if demands.is_some() { capacity } else { None };
        let load_id = if demands.is_some() { load_id } else { None };

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts,
            depot_location_fact,
            distances,
            node_count,
            load_id,
            capacity,
            demands,
        })
    }

    fn distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from * self.node_count + to]
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut propositional_values = vec![0usize; task.variables().len()];
        state.fill_state(registry, &mut propositional_values);

        let depot_holds = propositional_values[self.depot_location_fact.var_id]
            == self.depot_location_fact.value;

        let mut current_node: Option<usize> = if depot_holds { Some(0) } else { None };
        if current_node.is_none() {
            for (i, fact) in self.location_facts.iter().enumerate() {
                if propositional_values[fact.var_id] == fact.value {
                    current_node = Some(i + 1);
                    break;
                }
            }
        }

        let mut current = current_node.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "no current CVRP location found in state".to_string(),
            )
        })?;

        let mut remaining = vec![false; self.customer_names.len()];
        let mut remaining_count = 0usize;
        for (i, fact) in self.visited_facts.iter().enumerate() {
            if propositional_values[fact.var_id] != fact.value {
                remaining[i] = true;
                remaining_count += 1;
            }
        }

        if remaining_count == 0 {
            return Ok(self.distance(current, 0).max(0.0));
        }

        let mut current_load: f64 = 0.0;
        if let (Some(load_id), Some(_)) = (self.load_id, self.capacity) {
            let mut numeric_values = vec![0.0_f64; task.numeric_variables().len()];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
            current_load = numeric_values[load_id].max(0.0);
        }

        let mut h: f64 = 0.0;

        // Construct a complete continuation using nearest-neighbour routing.
        // With exact resource PNEs available, a route is closed through the
        // depot whenever no remaining customer fits in the current vehicle.
        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost: f64 = f64::INFINITY;

            for customer in 0..remaining.len() {
                if !remaining[customer] {
                    continue;
                }

                let feasible = match (&self.demands, self.capacity) {
                    (Some(demands), Some(capacity)) => {
                        current_load + demands[customer] <= capacity + 1.0e-9
                    }
                    _ => true,
                };

                if feasible {
                    let node = customer + 1;
                    let cost = self.distance(current, node);
                    if cost < best_cost {
                        best_cost = cost;
                        best_customer = Some(customer);
                    }
                }
            }

            if let Some(customer) = best_customer {
                h += best_cost;
                current = customer + 1;
                remaining[customer] = false;
                remaining_count -= 1;

                if let Some(demands) = &self.demands {
                    current_load += demands[customer];
                }
                continue;
            }

            // No customer fits. A visit-via-depot transition closes the
            // current route and starts a new one at a remaining customer.
            let mut next_customer: Option<usize> = None;
            let mut next_cost: f64 = f64::INFINITY;

            for customer in 0..remaining.len() {
                if !remaining[customer] {
                    continue;
                }

                let node = customer + 1;
                let cost = if current == 0 {
                    self.distance(0, node)
                } else {
                    self.distance(current, 0) + self.distance(0, node)
                };

                if cost < next_cost {
                    next_cost = cost;
                    next_customer = Some(customer);
                }
            }

            let customer = next_customer.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "failed to choose a remaining CVRP customer".to_string(),
                )
            })?;

            h += next_cost;
            current = customer + 1;
            remaining[customer] = false;
            remaining_count -= 1;
            current_load = self
                .demands
                .as_ref()
                .map(|demands| demands[customer])
                .unwrap_or(0.0);
        }

        h += self.distance(current, 0);

        if !h.is_finite() {
            return Err(EvaluationError::ComputationFailed(
                "non-finite CVRP heuristic value".to_string(),
            ));
        }

        Ok(h.max(0.0))
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
