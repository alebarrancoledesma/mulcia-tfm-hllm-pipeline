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

struct LlmHeuristic {
    // Node zero is the depot; the remaining nodes are customers.
    location_facts: Vec<ExplicitFact>,
    visited_facts: Vec<ExplicitFact>,
    distances: Vec<Vec<f64>>,

    numeric_variable_count: usize,
    load_id: Option<usize>,
    capacity: Option<f64>,
    demands: Vec<f64>,
    resources_complete: bool,
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

    let object = &text[prefix.len()..text.len() - 1];
    if object.is_empty() || object.contains(',') {
        None
    } else {
        Some(object.trim().to_string())
    }
}

fn clean_operator_token(token: &str) -> &str {
    token.trim_matches(|c| c == '(' || c == ')' || c == ',')
}

fn numeric_name_is(name: &str, exact: &str) -> bool {
    let trimmed = name.trim();
    trimmed == exact || trimmed == format!("PNE {}", exact)
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut visited_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        let mut location_by_name: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = parse_unary_atom(fact_name, "visited") {
                    if !visited_by_name.contains_key(&customer) {
                        customer_names.push(customer.clone());
                    }
                    visited_by_name.insert(customer, fact);
                } else if let Some(place) = parse_unary_atom(fact_name, "loc") {
                    location_by_name.insert(place, fact);
                }
            }
        }

        customer_names.sort();
        customer_names.dedup();

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP heuristic found no visited(customer) facts".to_string(),
            ));
        }

        let depot_name = location_by_name
            .keys()
            .find(|name| !visited_by_name.contains_key(*name))
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "CVRP heuristic could not identify the depot location fact"
                    .to_string()
                    .into()
            })?;

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());

        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let mut location_facts = Vec::with_capacity(node_names.len());
        for name in &node_names {
            let fact = location_by_name
                .get(name)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("missing loc({}) fact", name).into()
                })?;
            location_facts.push(fact);
        }

        // visited_facts[i] corresponds to node i + 1.
        let mut visited_facts = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let fact = visited_by_name
                .get(name)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("missing visited({}) fact", name).into()
                })?;
            visited_facts.push(fact);
        }

        let node_count = node_names.len();
        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for (i, row) in distances.iter_mut().enumerate() {
            row[i] = 0.0;
        }

        // Reconstruct the routing matrix from grounded action costs. This avoids
        // depending on translated travel-cost numeric-variable names.
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
                        distances[from][to] = distances[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = node_index.get(tokens[1]) {
                        distances[from][0] = distances[from][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        // A valid grounding supplies every depot/customer and customer/customer
        // arc through visit and return-to-depot operators.
        for i in 0..node_count {
            for j in 0..node_count {
                if i != j && !distances[i][j].is_finite() {
                    return Err(HeuristicBuildError::from(format!(
                        "CVRP heuristic could not reconstruct routing cost {} -> {}",
                        node_names[i], node_names[j]
                    )));
                }
            }
        }

        let numeric_variables = task.numeric_variables();
        let numeric_variable_count = numeric_variables.len();
        let initial_numeric = task.get_initial_numeric_state_values();

        let mut load_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;

        for (id, variable) in numeric_variables.iter().enumerate() {
            if numeric_name_is(variable.name(), "load()") {
                load_id = Some(id);
            }
            if numeric_name_is(variable.name(), "capacity()") {
                capacity_id = Some(id);
            }
        }

        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0);

        let mut demands = vec![0.0; customer_names.len()];
        let mut all_demands_found = true;

        for (customer_index, customer_name) in customer_names.iter().enumerate() {
            let exact = format!("demand({})", customer_name);
            let demand_id = numeric_variables
                .iter()
                .enumerate()
                .find(|(_, variable)| numeric_name_is(variable.name(), &exact))
                .map(|(id, _)| id);

            match demand_id.and_then(|id| initial_numeric.get(id).copied()) {
                Some(value) if value.is_finite() && value >= 0.0 => {
                    demands[customer_index] = value;
                }
                _ => {
                    all_demands_found = false;
                    demands[customer_index] = 0.0;
                }
            }
        }

        // Capacity and demands are optional because the translator may fold
        // them into derived expressions. Routing remains fully available when
        // that happens.
        let resources_complete =
            load_id.is_some() && capacity.is_some() && all_demands_found;

        Ok(Self {
            location_facts,
            visited_facts,
            distances,
            numeric_variable_count,
            load_id,
            capacity,
            demands,
            resources_complete,
        })
    }

    fn distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from][to]
    }

    fn greedy_completion(
        &self,
        current_node: usize,
        unvisited: &[bool],
        initial_load: f64,
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|value| **value).count();
        let mut current = current_node;
        let mut load = initial_load.max(0.0);
        let capacity = self.capacity.unwrap_or(f64::MAX);
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for customer_index in 0..remaining.len() {
                if !remaining[customer_index] {
                    continue;
                }

                let feasible = !self.resources_complete
                    || load + self.demands[customer_index] <= capacity + 1.0e-9;
                if !feasible {
                    continue;
                }

                let node = customer_index + 1;
                let cost = self.distance(current, node);
                if cost < best_cost {
                    best_cost = cost;
                    best_customer = Some(customer_index);
                }
            }

            if let Some(customer_index) = best_customer {
                let node = customer_index + 1;
                total += self.distance(current, node);
                current = node;
                load += self.demands[customer_index];
                remaining[customer_index] = false;
                remaining_count -= 1;
                continue;
            }

            // No remaining customer fits the current vehicle. Close this route
            // and continue from the depot with an empty load.
            if current != 0 {
                total += self.distance(current, 0);
                current = 0;
                load = 0.0;
                continue;
            }

            // Defensive fallback for malformed instances containing a demand
            // larger than capacity: select the nearest remaining customer so
            // evaluation always terminates.
            let mut fallback: Option<usize> = None;
            let mut fallback_cost = f64::INFINITY;
            for customer_index in 0..remaining.len() {
                if remaining[customer_index] {
                    let node = customer_index + 1;
                    let cost = self.distance(0, node);
                    if cost < fallback_cost {
                        fallback_cost = cost;
                        fallback = Some(customer_index);
                    }
                }
            }

            if let Some(customer_index) = fallback {
                let node = customer_index + 1;
                total += self.distance(0, node);
                current = node;
                load = self.demands[customer_index];
                remaining[customer_index] = false;
                remaining_count -= 1;
            } else {
                break;
            }
        }

        if current != 0 {
            total += self.distance(current, 0);
        }

        total
    }

    fn mst_bound(&self, active_nodes: &[usize]) -> f64 {
        if active_nodes.len() <= 1 {
            return 0.0;
        }

        let count = active_nodes.len();
        let mut in_tree = vec![false; count];
        let mut best_edge = vec![f64::INFINITY; count];
        best_edge[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;

            for i in 0..count {
                if !in_tree[i] && best_edge[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best_edge[i];
                }
            }

            let Some(selected_index) = selected else {
                break;
            };

            in_tree[selected_index] = true;
            total += selected_cost;
            let from = active_nodes[selected_index];

            for other_index in 0..count {
                if in_tree[other_index] {
                    continue;
                }
                let to = active_nodes[other_index];
                let undirected_cost = self
                    .distance(from, to)
                    .min(self.distance(to, from));
                if undirected_cost < best_edge[other_index] {
                    best_edge[other_index] = undirected_cost;
                }
            }
        }

        total
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

        let current_node = self
            .location_facts
            .iter()
            .position(|fact| fact.is_hold(state_view))
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "CVRP heuristic found no current location".to_string(),
                )
            })?;

        let mut unvisited = vec![false; self.visited_facts.len()];
        for (customer_index, fact) in self.visited_facts.iter().enumerate() {
            unvisited[customer_index] = !fact.is_hold(state_view);
        }

        let mut load: f64 = 0.0;
        if self.resources_complete {
            let mut numeric_values = vec![0.0; self.numeric_variable_count];
            state_view.fill_numeric(&mut numeric_values).map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

            if let Some(load_id) = self.load_id {
                load = numeric_values.get(load_id).copied().unwrap_or(0.0);
                if !load.is_finite() || load < 0.0 {
                    load = 0.0;
                }
            }
        }

        let greedy = self.greedy_completion(current_node, &unvisited, load);

        // The spanning structure contains the depot, the current endpoint, and
        // every unvisited customer. It supplies a stable global routing signal
        // that complements the stronger but more locally sensitive greedy tour.
        let mut active_nodes = Vec::with_capacity(unvisited.len() + 2);
        active_nodes.push(0);
        if current_node != 0 {
            active_nodes.push(current_node);
        }
        for (customer_index, is_unvisited) in unvisited.iter().enumerate() {
            if *is_unvisited {
                active_nodes.push(customer_index + 1);
            }
        }
        active_nodes.sort_unstable();
        active_nodes.dedup();

        let mst = self.mst_bound(&active_nodes);

        // A mostly constructive estimate gives strong search guidance, while
        // the MST component reduces nearest-neighbor myopia and instability.
        let h = (0.8 * greedy + 0.2 * mst).max(0.0);
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
