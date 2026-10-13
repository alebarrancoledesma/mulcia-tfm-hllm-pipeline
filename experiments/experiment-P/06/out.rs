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

use std::collections::{HashMap, HashSet};


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    depot_location_fact: ExplicitFact,
    depot_index: usize,
    distances: Vec<f64>,
    demands: Vec<f64>,
    capacity: Option<f64>,
    load_id: Option<usize>,
    numeric_variable_count: usize,
}

impl LlmHeuristic {
    fn atom_argument(name: &str, predicate: &str) -> Option<String> {
        let trimmed = name.trim();
        let prefix = format!("Atom {}(", predicate);
        if !trimmed.starts_with(&prefix) || !trimmed.ends_with(')') {
            return None;
        }

        let argument = &trimmed[prefix.len()..trimmed.len() - 1];
        if argument.is_empty() || argument.contains(',') {
            None
        } else {
            Some(argument.trim().to_string())
        }
    }

    fn operator_tokens(name: &str) -> Vec<&str> {
        name.trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split_whitespace()
            .collect()
    }

    fn is_exact_numeric_name(actual: &str, bare: &str) -> bool {
        let trimmed = actual.trim();
        trimmed == bare || trimmed == format!("PNE {}", bare)
    }

    fn distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from * (self.depot_index + 1) + to]
    }

    fn mst_connection_cost(&self, current: usize, unvisited: &[usize]) -> f64 {
        let mut active = Vec::with_capacity(unvisited.len() + 2);
        active.push(current);
        for &customer in unvisited {
            if !active.contains(&customer) {
                active.push(customer);
            }
        }
        if !active.contains(&self.depot_index) {
            active.push(self.depot_index);
        }

        if active.len() <= 1 {
            return 0.0;
        }

        let count = active.len();
        let mut used = vec![false; count];
        let mut best = vec![f64::INFINITY; count];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;
            for i in 0..count {
                if !used[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let Some(selected) = selected else {
                return 0.0;
            };
            if !selected_cost.is_finite() {
                return 0.0;
            }

            used[selected] = true;
            total += selected_cost;
            let from = active[selected];

            for j in 0..count {
                if used[j] {
                    continue;
                }
                let to = active[j];
                let edge = self.distance(from, to).min(self.distance(to, from));
                if edge < best[j] {
                    best[j] = edge;
                }
            }
        }

        total
    }

    fn greedy_completion_cost(
        &self,
        initial_location: usize,
        unvisited: &[usize],
        current_load: Option<f64>,
    ) -> f64 {
        let customer_count = self.customer_names.len();
        let mut remaining = vec![false; customer_count];
        for &customer in unvisited {
            remaining[customer] = true;
        }

        let use_capacity = self.capacity.is_some() && current_load.is_some();
        let capacity = self.capacity.unwrap_or(f64::MAX);
        let mut remaining_capacity = if use_capacity {
            if initial_location == self.depot_index {
                capacity
            } else {
                (capacity - current_load.unwrap_or(0.0)).max(0.0)
            }
        } else {
            f64::MAX
        };

        let mut current = initial_location;
        let mut remaining_count = unvisited.len();
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for customer in 0..customer_count {
                if !remaining[customer] {
                    continue;
                }
                if use_capacity && self.demands[customer] > remaining_capacity + 1.0e-9 {
                    continue;
                }

                let cost = self.distance(current, customer);
                if cost < best_cost {
                    best_cost = cost;
                    best_customer = Some(customer);
                }
            }

            if let Some(customer) = best_customer {
                if !best_cost.is_finite() {
                    break;
                }
                total += best_cost;
                current = customer;
                remaining[customer] = false;
                remaining_count -= 1;
                if use_capacity {
                    remaining_capacity =
                        (remaining_capacity - self.demands[customer]).max(0.0);
                }
                continue;
            }

            if use_capacity && current != self.depot_index {
                let return_cost = self.distance(current, self.depot_index);
                if !return_cost.is_finite() {
                    break;
                }
                total += return_cost;
                current = self.depot_index;
                remaining_capacity = capacity;
                continue;
            }

            // Permissive fallback for incomplete resource information or an
            // unexpectedly oversized demand: preserve routing guidance rather
            // than failing the heuristic evaluation.
            let mut fallback_customer: Option<usize> = None;
            let mut fallback_cost = f64::INFINITY;
            for customer in 0..customer_count {
                if remaining[customer] {
                    let cost = self.distance(current, customer);
                    if cost < fallback_cost {
                        fallback_cost = cost;
                        fallback_customer = Some(customer);
                    }
                }
            }

            let Some(customer) = fallback_customer else {
                break;
            };
            if !fallback_cost.is_finite() {
                break;
            }

            total += fallback_cost;
            current = customer;
            remaining[customer] = false;
            remaining_count -= 1;
            if use_capacity {
                remaining_capacity =
                    (capacity - self.demands[customer]).max(0.0);
            }
        }

        if current != self.depot_index {
            let return_cost = self.distance(current, self.depot_index);
            if return_cost.is_finite() {
                total += return_cost;
            }
        }

        total
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_set: HashSet<String> = HashSet::new();
        let mut depot_name: Option<String> = None;

        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            let fact_name = task.get_fact_name(goal);
            if let Some(customer) = Self::atom_argument(fact_name, "visited") {
                customer_set.insert(customer);
            } else if let Some(place) = Self::atom_argument(fact_name, "loc") {
                depot_name = Some(place);
            }
        }

        if customer_set.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 found no visited-customer goals".to_string(),
            ));
        }
        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not identify the depot location goal"
                .to_string()
                .into()
        })?;

        let mut customer_names: Vec<String> = customer_set.into_iter().collect();
        customer_names.sort();
        let customer_count = customer_names.len();
        let depot_index = customer_count;

        let mut customer_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            customer_index.insert(name.clone(), index);
        }

        let mut visited_options: Vec<Option<ExplicitFact>> =
            vec![None; customer_count];
        let mut location_options: Vec<Option<ExplicitFact>> =
            vec![None; customer_count];
        let mut depot_location: Option<ExplicitFact> = None;

        for variable_id in 0..task.variables().len() {
            let domain_size = task.variables()[variable_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = Self::atom_argument(fact_name, "visited") {
                    if let Some(&index) = customer_index.get(&customer) {
                        visited_options[index] = Some(fact);
                    }
                } else if let Some(place) = Self::atom_argument(fact_name, "loc") {
                    if place == depot_name {
                        depot_location = Some(fact);
                    } else if let Some(&index) = customer_index.get(&place) {
                        location_options[index] = Some(fact);
                    }
                }
            }
        }

        let mut visited_facts = Vec::with_capacity(customer_count);
        let mut location_facts = Vec::with_capacity(customer_count);
        for index in 0..customer_count {
            visited_facts.push(
                visited_options[index].take().ok_or_else(|| -> HeuristicBuildError {
                    format!(
                        "llm_001 could not find visited fact for {}",
                        customer_names[index]
                    )
                    .into()
                })?,
            );
            location_facts.push(
                location_options[index].take().ok_or_else(|| -> HeuristicBuildError {
                    format!(
                        "llm_001 could not find location fact for {}",
                        customer_names[index]
                    )
                    .into()
                })?,
            );
        }
        let depot_location_fact =
            depot_location.ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not find the depot location fact"
                    .to_string()
                    .into()
            })?;

        let node_count = customer_count + 1;
        let mut distances = vec![f64::INFINITY; node_count * node_count];
        for node in 0..node_count {
            distances[node * node_count + node] = 0.0;
        }

        let node_index = |name: &str| -> Option<usize> {
            if name == depot_name {
                Some(depot_index)
            } else {
                customer_index.get(name).copied()
            }
        };

        for operator in task.get_operators() {
            let tokens = Self::operator_tokens(operator.name());
            let metric_cost = metric_operator_cost_from_initial_values(task, operator);
            if !metric_cost.is_finite() || metric_cost < 0.0 {
                continue;
            }

            if tokens.len() == 3 && tokens[0] == "visit" {
                if let (Some(from), Some(to)) =
                    (node_index(tokens[1]), node_index(tokens[2]))
                {
                    let slot = from * node_count + to;
                    distances[slot] = distances[slot].min(metric_cost);
                }
            } else if tokens.len() == 2 && tokens[0] == "return-to-depot" {
                if let Some(from) = node_index(tokens[1]) {
                    let slot = from * node_count + depot_index;
                    distances[slot] = distances[slot].min(metric_cost);
                }
            }
        }

        for customer in 0..customer_count {
            if !distances[depot_index * node_count + customer].is_finite()
                || !distances[customer * node_count + depot_index].is_finite()
            {
                return Err(HeuristicBuildError::from(format!(
                    "llm_001 could not reconstruct depot routing costs for {}",
                    customer_names[customer]
                )));
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut capacity_id: Option<usize> = None;
        let mut load_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; customer_count];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if Self::is_exact_numeric_name(name, "capacity()") {
                capacity_id = Some(id);
            } else if Self::is_exact_numeric_name(name, "load()") {
                load_id = Some(id);
            } else {
                for (customer, customer_name) in customer_names.iter().enumerate() {
                    let exact = format!("demand({})", customer_name);
                    if Self::is_exact_numeric_name(name, &exact) {
                        demand_ids[customer] = Some(id);
                        break;
                    }
                }
            }
        }

        let all_demands_available = demand_ids.iter().all(Option::is_some);
        let resource_information_available =
            capacity_id.is_some() && load_id.is_some() && all_demands_available;

        let capacity = if resource_information_available {
            capacity_id.and_then(|id| initial_numeric.get(id).copied())
        } else {
            None
        };

        let mut demands = vec![0.0; customer_count];
        if resource_information_available {
            for customer in 0..customer_count {
                if let Some(id) = demand_ids[customer] {
                    demands[customer] = initial_numeric.get(id).copied().unwrap_or(0.0);
                }
            }
        }

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts,
            depot_location_fact,
            depot_index,
            distances,
            demands,
            capacity,
            load_id: if resource_information_available {
                load_id
            } else {
                None
            },
            numeric_variable_count: task.numeric_variables().len(),
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

        let mut current_location = self.depot_index;
        if !self.depot_location_fact.is_hold(state_view) {
            for (customer, fact) in self.location_facts.iter().enumerate() {
                if fact.is_hold(registry.view(state)) {
                    current_location = customer;
                    break;
                }
            }
        }

        let mut unvisited = Vec::with_capacity(self.customer_names.len());
        for (customer, fact) in self.visited_facts.iter().enumerate() {
            if !fact.is_hold(registry.view(state)) {
                unvisited.push(customer);
            }
        }

        let current_load = if let Some(load_id) = self.load_id {
            let mut numeric_values = vec![0.0; self.numeric_variable_count];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
            numeric_values.get(load_id).copied()
        } else {
            None
        };

        let greedy = self.greedy_completion_cost(
            current_location,
            &unvisited,
            current_load,
        );
        let connection = self.mst_connection_cost(current_location, &unvisited);

        // The greedy rollout supplies strong route-ordering guidance, while a
        // small MST component makes the estimate less sensitive to one greedy
        // choice and preserves a global view of the remaining customer set.
        let h = (0.9 * greedy + 0.1 * connection).max(0.0);
        Ok(if h.is_finite() { h } else { 0.0 })
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
