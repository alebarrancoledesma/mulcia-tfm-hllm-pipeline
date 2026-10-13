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

struct LlmHeuristic {
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    distances: Vec<Vec<f64>>,
    demands: Option<Vec<f64>>,
    capacity: Option<f64>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    max_vehicles: Option<f64>,
    numeric_variable_count: usize,
}

impl LlmHeuristic {
    fn atom_argument(name: &str, predicate: &str) -> Option<String> {
        let prefix = format!("Atom {}(", predicate);
        let trimmed = name.trim();
        if !trimmed.starts_with(&prefix) || !trimmed.ends_with(')') {
            return None;
        }

        let argument = &trimmed[prefix.len()..trimmed.len() - 1];
        let argument = argument.trim();
        if argument.is_empty() || argument.contains(',') {
            None
        } else {
            Some(argument.to_string())
        }
    }

    fn normalized_numeric_name(name: &str) -> &str {
        let trimmed = name.trim();
        trimmed.strip_prefix("PNE ").unwrap_or(trimmed).trim()
    }

    fn demand_argument(name: &str) -> Option<String> {
        let normalized = Self::normalized_numeric_name(name);
        let prefix = "demand(";
        if !normalized.starts_with(prefix) || !normalized.ends_with(')') {
            return None;
        }

        let argument = normalized[prefix.len()..normalized.len() - 1].trim();
        if argument.is_empty() || argument.contains(',') {
            None
        } else {
            Some(argument.to_string())
        }
    }

    fn clean_operator_token(token: &str) -> &str {
        token.trim_matches(|c: char| {
            c == '(' || c == ')' || c == ',' || c.is_whitespace()
        })
    }

    fn symmetric_distance(&self, a: usize, b: usize) -> f64 {
        self.distances[a][b].min(self.distances[b][a])
    }

    fn mst_cost(&self, nodes: &[usize]) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let count = nodes.len();
        let mut selected = vec![false; count];
        let mut cheapest = vec![f64::INFINITY; count];
        cheapest[0] = 0.0;

        let mut total: f64 = 0.0;
        for _ in 0..count {
            let mut best_position: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for position in 0..count {
                if !selected[position] && cheapest[position] < best_cost {
                    best_cost = cheapest[position];
                    best_position = Some(position);
                }
            }

            let Some(position) = best_position else {
                return f64::INFINITY;
            };

            selected[position] = true;
            total += best_cost;

            for other in 0..count {
                if !selected[other] {
                    let edge = self.symmetric_distance(nodes[position], nodes[other]);
                    if edge < cheapest[other] {
                        cheapest[other] = edge;
                    }
                }
            }
        }

        total
    }

    fn greedy_completion_cost(
        &self,
        start: usize,
        unvisited: &[usize],
        initial_load: f64,
    ) -> f64 {
        let depot = 0usize;
        let mut remaining = unvisited.to_vec();
        let mut current = start;
        let mut total: f64 = 0.0;

        let resource_data = match (&self.demands, self.capacity) {
            (Some(demands), Some(capacity)) if capacity.is_finite() && capacity > 0.0 => {
                Some((demands, capacity))
            }
            _ => None,
        };

        let mut available = if let Some((_, capacity)) = resource_data {
            (capacity - initial_load).max(0.0)
        } else {
            f64::INFINITY
        };

        while !remaining.is_empty() {
            let mut best_position: Option<usize> = None;
            let mut best_score = f64::INFINITY;

            for (position, &customer) in remaining.iter().enumerate() {
                if let Some((demands, _)) = resource_data {
                    if demands[customer] > available + 1.0e-9 {
                        continue;
                    }
                }

                // The small depot-distance term avoids leaving geographically
                // isolated customers as expensive final endpoints.
                let score = self.distances[current][customer]
                    + 0.10 * self.distances[customer][depot];
                if score < best_score {
                    best_score = score;
                    best_position = Some(position);
                }
            }

            if let Some(position) = best_position {
                let customer = remaining.swap_remove(position);
                total += self.distances[current][customer];
                current = customer;

                if let Some((demands, _)) = resource_data {
                    available = (available - demands[customer]).max(0.0);
                }
            } else if let Some((demands, capacity)) = resource_data {
                // No customer fits in the active vehicle. Close this route and
                // start a fresh route through visit-via-depot.
                if current == depot {
                    // A demand larger than capacity makes the state unsolvable.
                    if remaining.iter().any(|&customer| {
                        demands[customer] > capacity + 1.0e-9
                    }) {
                        return f64::INFINITY;
                    }
                    return f64::INFINITY;
                }

                total += self.distances[current][depot];
                current = depot;
                available = capacity;
            } else {
                return f64::INFINITY;
            }
        }

        if current != depot {
            total += self.distances[current][depot];
        }

        total
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut positive_visited: HashMap<String, ExplicitFact> = HashMap::new();
        let mut positive_locations: HashMap<String, ExplicitFact> = HashMap::new();

        for variable_id in 0..task.variables().len() {
            let domain_size = task.variables()[variable_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = Self::atom_argument(fact_name, "visited") {
                    positive_visited.insert(object, fact);
                } else if let Some(object) = Self::atom_argument(fact_name, "loc") {
                    positive_locations.insert(object, fact);
                }
            }
        }

        if positive_visited.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP heuristic found no positive visited facts".to_string(),
            ));
        }

        let mut customer_names: Vec<String> = positive_visited.keys().cloned().collect();
        customer_names.sort();

        let depot_candidates: Vec<String> = positive_locations
            .keys()
            .filter(|name| !positive_visited.contains_key(*name))
            .cloned()
            .collect();

        if depot_candidates.len() != 1 {
            return Err(HeuristicBuildError::from(
                "CVRP heuristic could not uniquely identify the depot location"
                    .to_string(),
            ));
        }
        let depot_name = depot_candidates[0].clone();

        let depot_fact = positive_locations.get(&depot_name).cloned().ok_or_else(
            || -> HeuristicBuildError {
                "missing positive depot location fact".to_string().into()
            },
        )?;

        // Node zero is the depot. Every other node corresponds to one customer.
        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());

        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let mut visited_facts = Vec::with_capacity(node_names.len());
        // The depot has no visited predicate; this entry is never queried.
        visited_facts.push(depot_fact.clone());

        let mut location_facts = Vec::with_capacity(node_names.len());
        location_facts.push(depot_fact);

        for customer in &customer_names {
            let visited = positive_visited.get(customer).cloned().ok_or_else(
                || -> HeuristicBuildError {
                    "missing customer visited fact".to_string().into()
                },
            )?;
            let location = positive_locations.get(customer).cloned().ok_or_else(
                || -> HeuristicBuildError {
                    "missing customer location fact".to_string().into()
                },
            )?;
            visited_facts.push(visited);
            location_facts.push(location);
        }

        let node_count = node_names.len();
        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for (index, row) in distances.iter_mut().enumerate() {
            row[index] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens: Vec<&str> = operator
                .name()
                .split_whitespace()
                .map(Self::clean_operator_token)
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

        // If a direct grounded arc is unexpectedly absent, its route through
        // the depot is still a valid routing estimate whenever both legs exist.
        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !distances[from][to].is_finite() {
                    let via_depot = distances[from][0] + distances[0][to];
                    if via_depot.is_finite() {
                        distances[from][to] = via_depot;
                    }
                }
            }
        }

        for from in 0..node_count {
            for to in 0..node_count {
                if !distances[from][to].is_finite() {
                    return Err(HeuristicBuildError::from(
                        "CVRP heuristic could not reconstruct a complete routing matrix"
                            .to_string(),
                    ));
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut demand_by_name: HashMap<String, f64> = HashMap::new();

        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            let normalized = Self::normalized_numeric_name(variable.name());
            if normalized == "load()" {
                load_id = Some(numeric_id);
            } else if normalized == "vehicles()" {
                vehicles_id = Some(numeric_id);
            } else if normalized == "capacity()" {
                capacity = initial_numeric.get(numeric_id).copied();
            } else if normalized == "max_vehicles()" {
                max_vehicles = initial_numeric.get(numeric_id).copied();
            } else if let Some(customer) = Self::demand_argument(variable.name()) {
                if let Some(value) = initial_numeric.get(numeric_id) {
                    demand_by_name.insert(customer, *value);
                }
            }
        }

        // Capacity and demands are deliberately optional: the translator may
        // fold them into derived comparison expressions.
        let demands = if capacity.is_some()
            && customer_names
                .iter()
                .all(|customer| demand_by_name.contains_key(customer))
        {
            let mut values = vec![0.0; node_count];
            for (offset, customer) in customer_names.iter().enumerate() {
                values[offset + 1] = demand_by_name[customer];
            }
            Some(values)
        } else {
            None
        };

        Ok(Self {
            visited_facts,
            location_facts,
            distances,
            demands,
            capacity,
            load_id,
            vehicles_id,
            max_vehicles,
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
        let view = registry.view(state);

        let mut current: Option<usize> = None;
        for (node, fact) in self.location_facts.iter().enumerate() {
            if fact.is_hold(view) {
                current = Some(node);
                break;
            }
        }

        let current = current.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "CVRP heuristic found no current location".to_string(),
            )
        })?;

        let mut unvisited = Vec::with_capacity(self.visited_facts.len());
        for customer in 1..self.visited_facts.len() {
            if !self.visited_facts[customer].is_hold(registry.view(state)) {
                unvisited.push(customer);
            }
        }

        // Once every customer is visited, only the mandatory return action
        // remains. This exact terminal estimate is especially informative.
        if unvisited.is_empty() {
            return Ok(self.distances[current][0].max(0.0));
        }

        let resource_complete = self.demands.is_some()
            && self.capacity.is_some()
            && self.load_id.is_some();

        let mut current_load: f64 = 0.0;
        let mut current_vehicles: Option<f64> = None;

        if resource_complete
            || (self.vehicles_id.is_some() && self.max_vehicles.is_some())
        {
            let mut numeric_values = vec![0.0; self.numeric_variable_count];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            if let Some(load_id) = self.load_id {
                current_load = numeric_values[load_id];
            }
            if let Some(vehicles_id) = self.vehicles_id {
                current_vehicles = Some(numeric_values[vehicles_id]);
            }
        }

        // Detect a proven vehicle-limit dead end only when all required exact
        // resource signals survived translation.
        if let (
            Some(demands),
            Some(capacity),
            Some(vehicles),
            Some(max_vehicles),
        ) = (
            self.demands.as_ref(),
            self.capacity,
            current_vehicles,
            self.max_vehicles,
        ) {
            if capacity > 0.0 {
                let remaining_demand: f64 =
                    unvisited.iter().map(|&node| demands[node]).sum();
                let active_space = (capacity - current_load).max(0.0);
                let demand_after_active = (remaining_demand - active_space).max(0.0);
                let additional_vehicles =
                    (demand_after_active / capacity - 1.0e-12).ceil().max(0.0);

                if vehicles + additional_vehicles > max_vehicles + 1.0e-9 {
                    return Ok(f64::INFINITY);
                }
            }
        }

        // MST relaxation over the current endpoint, unvisited customers, and
        // depot supplies a stable global connectivity estimate.
        let mut mst_nodes = Vec::with_capacity(unvisited.len() + 2);
        mst_nodes.push(current);
        mst_nodes.extend(unvisited.iter().copied());
        if current != 0 {
            mst_nodes.push(0);
        }
        let mst = self.mst_cost(&mst_nodes);

        // A fast nearest-neighbour completion estimates ordering and, whenever
        // exact resource PNEs are available, inserts capacity-induced depot
        // route breaks.
        let greedy = self.greedy_completion_cost(
            current,
            &unvisited,
            if resource_complete { current_load } else { 0.0 },
        );

        let h = if mst.is_finite() && greedy.is_finite() {
            0.30 * mst + 0.70 * greedy
        } else if greedy.is_finite() {
            greedy
        } else if mst.is_finite() {
            mst
        } else {
            f64::INFINITY
        };

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
