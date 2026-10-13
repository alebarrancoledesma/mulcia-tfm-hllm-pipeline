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


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

struct Scratch {
    nodes: Vec<usize>,
    remaining: Vec<bool>,
    key: Vec<f64>,
    used: Vec<bool>,
    numeric_values: Vec<f64>,
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    distances: Vec<f64>,
    node_count: usize,
    depot: usize,
    demands: Vec<f64>,
    capacity: Option<f64>,
    load_id: Option<usize>,
    resources_available: bool,
    scratch: RefCell<Scratch>,
}

fn predicate_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let trimmed = name.trim();
    let body = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);
    body.strip_prefix(&prefix)?.strip_suffix(')')
}

fn normalized_operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_matches(|c| c == '(' || c == ')')
        .split_whitespace()
        .collect()
}

fn numeric_name_is(actual: &str, expected: &str) -> bool {
    let actual = actual.trim();
    actual == expected
        || actual
            .strip_prefix("PNE ")
            .map(|rest| rest.trim() == expected)
            .unwrap_or(false)
}

fn customer_index(names: &[String], name: &str) -> Option<usize> {
    names.iter().position(|candidate| candidate == name)
}

fn place_index(names: &[String], depot_name: &str, name: &str) -> Option<usize> {
    if name == depot_name {
        Some(names.len())
    } else {
        customer_index(names, name)
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customers_with_facts: Vec<(String, ExplicitFact)> = Vec::new();
        let mut depot_name: Option<String> = None;

        for goal_id in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_id);
            let fact_name = task.get_fact_name(&fact);

            if let Some(customer) = predicate_argument(fact_name, "visited") {
                customers_with_facts.push((customer.to_string(), fact));
            } else if let Some(place) = predicate_argument(fact_name, "loc") {
                depot_name = Some(place.to_string());
            }
        }

        if customers_with_facts.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not find CVRP visited goals".to_string(),
            ));
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not find the depot location goal"
                .to_string()
                .into()
        })?;

        customers_with_facts.sort_by(|a, b| a.0.cmp(&b.0));
        customers_with_facts.dedup_by(|a, b| a.0 == b.0);

        let customer_names: Vec<String> = customers_with_facts
            .iter()
            .map(|entry| entry.0.clone())
            .collect();
        let visited_facts: Vec<ExplicitFact> = customers_with_facts
            .into_iter()
            .map(|entry| entry.1)
            .collect();

        let depot = customer_names.len();
        let node_count = depot + 1;
        let mut location_facts: Vec<Option<ExplicitFact>> = vec![None; node_count];

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(place) = predicate_argument(fact_name, "loc") {
                    if place == depot_name {
                        location_facts[depot] = Some(fact);
                    } else if let Some(index) = customer_index(&customer_names, place) {
                        location_facts[index] = Some(fact);
                    }
                }
            }
        }

        let mut resolved_locations: Vec<ExplicitFact> = Vec::with_capacity(node_count);
        for (index, fact) in location_facts.into_iter().enumerate() {
            match fact {
                Some(value) => resolved_locations.push(value),
                None => {
                    return Err(HeuristicBuildError::from(format!(
                        "llm_001 could not resolve location fact for node {}",
                        index
                    )));
                }
            }
        }

        let mut distances = vec![f64::INFINITY; node_count * node_count];
        for index in 0..node_count {
            distances[index * node_count + index] = 0.0;
        }

        for op in task.get_operators() {
            let tokens = normalized_operator_tokens(op.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    let from = place_index(&customer_names, &depot_name, tokens[1]);
                    let to = customer_index(&customer_names, tokens[2]);
                    if let (Some(from), Some(to)) = (from, to) {
                        let slot = from * node_count + to;
                        distances[slot] = distances[slot].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(from) = customer_index(&customer_names, tokens[1]) {
                        let slot = from * node_count + depot;
                        distances[slot] = distances[slot].min(cost);
                    }
                }
                _ => {}
            }
        }

        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !distances[from * node_count + to].is_finite() {
                    return Err(HeuristicBuildError::from(format!(
                        "llm_001 could not reconstruct routing cost from node {} to node {}",
                        from, to
                    )));
                }
            }
        }

        let mut capacity_id: Option<usize> = None;
        let mut load_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; customer_names.len()];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if numeric_name_is(name, "capacity()") {
                capacity_id = Some(id);
            } else if numeric_name_is(name, "load()") {
                load_id = Some(id);
            } else {
                for (customer, customer_name) in customer_names.iter().enumerate() {
                    let expected = format!("demand({})", customer_name);
                    if numeric_name_is(name, &expected) {
                        demand_ids[customer] = Some(id);
                        break;
                    }
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0);

        let all_demands_available = demand_ids.iter().all(|id| id.is_some());
        let mut demands = vec![0.0; customer_names.len()];
        if all_demands_available {
            for customer in 0..customer_names.len() {
                if let Some(id) = demand_ids[customer] {
                    demands[customer] = initial_numeric
                        .get(id)
                        .copied()
                        .filter(|value| value.is_finite() && *value >= 0.0)
                        .unwrap_or(0.0);
                }
            }
        }

        let resources_available = capacity.is_some()
            && load_id.is_some()
            && all_demands_available
            && demands.iter().all(|demand| demand.is_finite());

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts: resolved_locations,
            distances,
            node_count,
            depot,
            demands,
            capacity,
            load_id,
            resources_available,
            scratch: RefCell::new(Scratch {
                nodes: Vec::with_capacity(node_count),
                remaining: vec![false; depot],
                key: vec![f64::INFINITY; node_count],
                used: vec![false; node_count],
                numeric_values: vec![0.0; task.numeric_variables().len()],
            }),
        })
    }

    fn directed_distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from * self.node_count + to]
    }

    fn undirected_distance(&self, first: usize, second: usize) -> f64 {
        self.directed_distance(first, second)
            .min(self.directed_distance(second, first))
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
        let mut scratch = self.scratch.borrow_mut();

        let current = self
            .location_facts
            .iter()
            .position(|fact| fact.is_hold(view))
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "llm_001 could not determine the current location".to_string(),
                )
            })?;

        scratch.nodes.clear();
        scratch.nodes.push(current);
        if current != self.depot {
            scratch.nodes.push(self.depot);
        }

        for customer in 0..self.customer_names.len() {
            let unvisited = !self.visited_facts[customer].is_hold(registry.view(state));
            scratch.remaining[customer] = unvisited;
            if unvisited && customer != current {
                scratch.nodes.push(customer);
            }
        }

        if scratch.nodes.len() == 1 && current == self.depot {
            return Ok(0.0);
        }

        let active_nodes = scratch.nodes.len();
        for index in 0..active_nodes {
            scratch.key[index] = f64::INFINITY;
            scratch.used[index] = false;
        }
        scratch.key[0] = 0.0;

        let mut mst_cost: f64 = 0.0;
        for _ in 0..active_nodes {
            let mut best_index: Option<usize> = None;
            let mut best_key: f64 = f64::INFINITY;
            for index in 0..active_nodes {
                if !scratch.used[index] && scratch.key[index] < best_key {
                    best_key = scratch.key[index];
                    best_index = Some(index);
                }
            }

            let selected = best_index.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "llm_001 encountered a disconnected routing graph".to_string(),
                )
            })?;
            scratch.used[selected] = true;
            mst_cost += scratch.key[selected];

            let selected_node = scratch.nodes[selected];
            for other in 0..active_nodes {
                if !scratch.used[other] {
                    let edge = self.undirected_distance(
                        selected_node,
                        scratch.nodes[other],
                    );
                    if edge < scratch.key[other] {
                        scratch.key[other] = edge;
                    }
                }
            }
        }

        let mut remaining_capacity: f64 = f64::INFINITY;
        let capacity = self.capacity.unwrap_or(f64::MAX);
        if self.resources_available {
            scratch
                .numeric_values
                .resize(eval_state.task().numeric_variables().len(), 0.0);
            registry
                .view(state)
                .fill_numeric(&mut scratch.numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            if let Some(load_id) = self.load_id {
                let load = scratch.numeric_values[load_id];
                remaining_capacity = (capacity - load).max(0.0);
            }
        }

        let mut rollout_cost: f64 = 0.0;
        let mut rollout_location = current;
        let mut customers_left = scratch.remaining.iter().filter(|value| **value).count();

        while customers_left > 0 {
            let mut best_feasible: Option<usize> = None;
            let mut best_feasible_cost: f64 = f64::INFINITY;

            for customer in 0..self.customer_names.len() {
                if !scratch.remaining[customer] {
                    continue;
                }
                let demand = if self.resources_available {
                    self.demands[customer]
                } else {
                    0.0
                };
                if demand <= remaining_capacity + 1.0e-9 {
                    let cost = self.directed_distance(rollout_location, customer);
                    if cost < best_feasible_cost {
                        best_feasible_cost = cost;
                        best_feasible = Some(customer);
                    }
                }
            }

            let selected = if let Some(customer) = best_feasible {
                rollout_cost += best_feasible_cost;
                customer
            } else {
                let mut best_restart: Option<usize> = None;
                let mut best_restart_cost: f64 = f64::INFINITY;
                for customer in 0..self.customer_names.len() {
                    if scratch.remaining[customer] {
                        let cost = self.directed_distance(rollout_location, self.depot)
                            + self.directed_distance(self.depot, customer);
                        if cost < best_restart_cost {
                            best_restart_cost = cost;
                            best_restart = Some(customer);
                        }
                    }
                }

                let customer = best_restart.ok_or_else(|| {
                    EvaluationError::ComputationFailed(
                        "llm_001 failed to select an unvisited customer".to_string(),
                    )
                })?;
                rollout_cost += best_restart_cost;
                remaining_capacity = capacity;
                customer
            };

            if self.resources_available {
                remaining_capacity =
                    (remaining_capacity - self.demands[selected]).max(0.0);
            }
            scratch.remaining[selected] = false;
            customers_left -= 1;
            rollout_location = selected;
        }

        rollout_cost += self.directed_distance(rollout_location, self.depot);

        // The MST supplies a stable global connectivity estimate, while the
        // capacity-aware nearest-neighbour rollout estimates an actual
        // completion. Blending reduces rollout overestimation while retaining
        // considerably more guidance than the MST alone.
        let blended = 0.85_f64 * rollout_cost + 0.15_f64 * mst_cost;
        let h = mst_cost.max(blended).max(0.0);

        Ok(h)
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
