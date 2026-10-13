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

struct ResourceInfo {
    load_id: usize,
    capacity: f64,
    demands: Vec<f64>,
}

struct LlmHeuristic {
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    distances: Vec<Vec<f64>>,
    resource: Option<ResourceInfo>,
}

fn atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let name = name.trim();
    let prefix = format!("Atom {}(", predicate);
    name.strip_prefix(&prefix)?.strip_suffix(')')
}

fn normalized_operator_tokens(name: &str) -> Vec<String> {
    name.trim()
        .trim_matches(|c| c == '(' || c == ')')
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c| c == '(' || c == ')' || c == ',')
                .to_string()
        })
        .collect()
}

fn normalized_numeric_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed)
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        for i in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(i);
            let fact_name = task.get_fact_name(&goal);

            if let Some(customer) = atom_argument(fact_name, "visited") {
                customer_names.push(customer.to_string());
            } else if let Some(place) = atom_argument(fact_name, "loc") {
                depot_name = Some(place.to_string());
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP task has no visited-customer goals".to_string(),
            ));
        }

        let mut customer_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            customer_index.insert(name.clone(), index + 1);
        }

        let node_count = customer_names.len() + 1;
        let mut visited_facts: Vec<Option<ExplicitFact>> = vec![None; customer_names.len()];
        let mut location_facts: Vec<Option<ExplicitFact>> = vec![None; node_count];

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = atom_argument(fact_name, "visited") {
                    if let Some(&node) = customer_index.get(customer) {
                        visited_facts[node - 1] = Some(fact.clone());
                    }
                }

                if let Some(place) = atom_argument(fact_name, "loc") {
                    if place == depot_name {
                        location_facts[0] = Some(fact.clone());
                    } else if let Some(&node) = customer_index.get(place) {
                        location_facts[node] = Some(fact.clone());
                    }
                }
            }
        }

        let mut final_visited_facts = Vec::with_capacity(customer_names.len());
        for fact in visited_facts {
            final_visited_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                "missing a visited-customer propositional fact".to_string().into()
            })?);
        }

        let mut final_location_facts = Vec::with_capacity(node_count);
        for fact in location_facts {
            final_location_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                "missing a CVRP location propositional fact".to_string().into()
            })?);
        }

        let mut node_index: HashMap<String, usize> = HashMap::new();
        node_index.insert(depot_name.clone(), 0);
        for (index, name) in customer_names.iter().enumerate() {
            node_index.insert(name.clone(), index + 1);
        }

        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for (i, row) in distances.iter_mut().enumerate() {
            row[i] = 0.0;
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

            match tokens[0].as_str() {
                "visit" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(&tokens[1]), node_index.get(&tokens[2]))
                    {
                        distances[from][to] = distances[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = node_index.get(&tokens[1]) {
                        distances[from][0] = distances[from][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !distances[from][to].is_finite() {
                    return Err(HeuristicBuildError::from(format!(
                        "could not reconstruct routing cost from node {} to node {}",
                        from, to
                    )));
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; customer_names.len()];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = normalized_numeric_name(variable.name());

            if name == "load()" {
                load_id = Some(id);
            } else if name == "capacity()" {
                capacity_id = Some(id);
            } else {
                for (customer, demand_id) in customer_names.iter().zip(demand_ids.iter_mut()) {
                    let expected = format!("demand({})", customer);
                    if name == expected {
                        *demand_id = Some(id);
                        break;
                    }
                }
            }
        }

        let resource = if let (Some(load_id), Some(capacity_id)) = (load_id, capacity_id) {
            if demand_ids.iter().all(Option::is_some)
                && load_id < initial_numeric.len()
                && capacity_id < initial_numeric.len()
            {
                let capacity = initial_numeric[capacity_id];
                let mut demands = Vec::with_capacity(customer_names.len());
                let mut valid = capacity.is_finite() && capacity > 0.0;

                for demand_id in demand_ids {
                    let id = demand_id.unwrap();
                    if id >= initial_numeric.len() {
                        valid = false;
                        break;
                    }
                    let demand = initial_numeric[id];
                    if !demand.is_finite() || demand < 0.0 || demand > capacity {
                        valid = false;
                        break;
                    }
                    demands.push(demand);
                }

                if valid && demands.len() == customer_names.len() {
                    Some(ResourceInfo {
                        load_id,
                        capacity,
                        demands,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            visited_facts: final_visited_facts,
            location_facts: final_location_facts,
            distances,
            resource,
        })
    }

    fn undirected_distance(&self, a: usize, b: usize) -> f64 {
        self.distances[a][b].min(self.distances[b][a])
    }

    fn mst_estimate(&self, current: usize, unvisited: &[bool]) -> f64 {
        let node_count = self.distances.len();
        let mut active = vec![false; node_count];
        active[0] = true;
        active[current] = true;
        for customer in 1..node_count {
            if unvisited[customer - 1] {
                active[customer] = true;
            }
        }

        let active_count = active.iter().filter(|&&value| value).count();
        if active_count <= 1 {
            return 0.0;
        }

        let mut in_tree = vec![false; node_count];
        let mut best = vec![f64::INFINITY; node_count];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..active_count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;

            for node in 0..node_count {
                if active[node] && !in_tree[node] && best[node] < selected_cost {
                    selected = Some(node);
                    selected_cost = best[node];
                }
            }

            let Some(node) = selected else {
                return 0.0;
            };

            in_tree[node] = true;
            total += selected_cost;

            for other in 0..node_count {
                if active[other] && !in_tree[other] {
                    let edge = self.undirected_distance(node, other);
                    if edge < best[other] {
                        best[other] = edge;
                    }
                }
            }
        }

        total
    }

    fn cheapest_insertion_estimate(&self, current: usize, unvisited: &[bool]) -> f64 {
        let mut route = vec![current, 0];

        for customer in 1..self.distances.len() {
            if !unvisited[customer - 1] {
                continue;
            }

            let mut best_position = 1usize;
            let mut best_delta = f64::INFINITY;

            for position in 0..route.len() - 1 {
                let from = route[position];
                let to = route[position + 1];
                let delta = self.distances[from][customer]
                    + self.distances[customer][to]
                    - self.distances[from][to];

                if delta < best_delta {
                    best_delta = delta;
                    best_position = position + 1;
                }
            }

            route.insert(best_position, customer);
        }

        let mut total: f64 = 0.0;
        for edge in route.windows(2) {
            total += self.distances[edge[0]][edge[1]];
        }
        total
    }

    fn capacity_greedy_estimate(
        &self,
        current: usize,
        unvisited: &[bool],
        current_load: f64,
        resource: &ResourceInfo,
    ) -> Option<f64> {
        if !current_load.is_finite() || current_load < 0.0 {
            return None;
        }

        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|&&value| value).count();
        let mut position = current;
        let mut capacity_left = if current == 0 {
            resource.capacity
        } else {
            (resource.capacity - current_load).max(0.0)
        };
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for customer in 1..self.distances.len() {
                if remaining[customer - 1]
                    && resource.demands[customer - 1] <= capacity_left + 1.0e-9
                    && self.distances[position][customer] < best_cost
                {
                    best_customer = Some(customer);
                    best_cost = self.distances[position][customer];
                }
            }

            if let Some(customer) = best_customer {
                total += best_cost;
                capacity_left -= resource.demands[customer - 1];
                position = customer;
                remaining[customer - 1] = false;
                remaining_count -= 1;
            } else {
                if position == 0 {
                    return None;
                }
                total += self.distances[position][0];
                position = 0;
                capacity_left = resource.capacity;
            }
        }

        total += self.distances[position][0];
        Some(total)
    }

    fn radial_resource_estimate(&self, unvisited: &[bool], resource: &ResourceInfo) -> f64 {
        let mut total: f64 = 0.0;
        for customer in 1..self.distances.len() {
            if unvisited[customer - 1] {
                let round_trip = self.distances[0][customer] + self.distances[customer][0];
                total += round_trip * resource.demands[customer - 1] / resource.capacity;
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let view = registry.view(state);

        let mut current = 0usize;
        for (node, fact) in self.location_facts.iter().enumerate() {
            if fact.is_hold(view) {
                current = node;
                break;
            }
        }

        let mut unvisited = vec![false; self.visited_facts.len()];
        for (index, fact) in self.visited_facts.iter().enumerate() {
            unvisited[index] = !fact.is_hold(view);
        }

        let mst = self.mst_estimate(current, &unvisited);
        let insertion = self.cheapest_insertion_estimate(current, &unvisited);

        // Blend a robust connectivity bound with the more discriminating
        // cheapest-insertion completion estimate.
        let mut h: f64 = 0.25 * mst + 0.75 * insertion;
        h = h.max(mst);

        if let Some(resource) = &self.resource {
            let mut numeric_values = vec![0.0; task.numeric_variables().len()];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            if resource.load_id < numeric_values.len() {
                if let Some(capacity_greedy) = self.capacity_greedy_estimate(
                    current,
                    &unvisited,
                    numeric_values[resource.load_id],
                    resource,
                ) {
                    // Multiple depot trips are often required in CVRP. This
                    // blend raises the single-tour estimate without making it
                    // entirely dependent on one greedy packing order.
                    let resource_blend = 0.60 * insertion + 0.40 * capacity_greedy;
                    h = h.max(resource_blend);
                }
            }

            h = h.max(self.radial_resource_estimate(&unvisited, resource));
        }

        if !h.is_finite() || h < 0.0 {
            Ok(0.0)
        } else {
            Ok(h)
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
