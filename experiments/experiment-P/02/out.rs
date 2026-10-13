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

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let trimmed = name.trim();
    let atom = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);

    if !atom.starts_with(&prefix) || !atom.ends_with(')') {
        return None;
    }

    let argument = &atom[prefix.len()..atom.len() - 1];
    if argument.is_empty() || argument.contains(',') {
        return None;
    }

    Some(argument.trim().to_string())
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

fn exact_numeric_id(task: &dyn AbstractNumericTask, base_name: &str) -> Option<usize> {
    let with_prefix = format!("PNE {}", base_name);

    task.numeric_variables()
        .iter()
        .enumerate()
        .find_map(|(id, variable)| {
            let name = variable.name().trim();
            if name == base_name || name == with_prefix {
                Some(id)
            } else {
                None
            }
        })
}

struct LlmHeuristic {
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    distances: Vec<f64>,
    node_count: usize,

    numeric_count: usize,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    demands: Vec<f64>,
    resource_information_complete: bool,

    dead_end_penalty: f64,
}

impl LlmHeuristic {
    fn distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from * self.node_count + to]
    }

    fn undirected_distance(&self, a: usize, b: usize) -> f64 {
        self.distance(a, b).min(self.distance(b, a))
    }

    fn routing_lower_bound(&self, current: usize, unvisited: &[bool]) -> f64 {
        let mut active: Vec<usize> = Vec::with_capacity(self.node_count + 1);
        active.push(0);

        if current != 0 {
            active.push(current);
        }

        for customer in 1..self.node_count {
            if unvisited[customer] && customer != current {
                active.push(customer);
            }
        }

        if active.len() <= 1 {
            return 0.0;
        }

        // Prim MST on the depot, current position, and all unvisited customers.
        let mut in_tree = vec![false; active.len()];
        let mut best = vec![f64::INFINITY; active.len()];
        best[0] = 0.0;
        let mut mst: f64 = 0.0;

        for _ in 0..active.len() {
            let mut chosen: Option<usize> = None;
            let mut chosen_cost = f64::INFINITY;

            for i in 0..active.len() {
                if !in_tree[i] && best[i] < chosen_cost {
                    chosen = Some(i);
                    chosen_cost = best[i];
                }
            }

            let Some(u) = chosen else {
                break;
            };

            in_tree[u] = true;
            mst += chosen_cost;

            for v in 0..active.len() {
                if !in_tree[v] {
                    let edge = self.undirected_distance(active[u], active[v]);
                    if edge < best[v] {
                        best[v] = edge;
                    }
                }
            }
        }

        // Degree relaxation for a path from the current position to the depot.
        // If already at the depot, the relaxed completion is a depot cycle.
        let mut degree_sum: f64 = 0.0;
        for &node in &active {
            let required_degree = if current == 0 {
                2
            } else if node == 0 || node == current {
                1
            } else {
                2
            };

            let mut first = f64::INFINITY;
            let mut second = f64::INFINITY;

            for &other in &active {
                if other == node {
                    continue;
                }
                let edge = self.undirected_distance(node, other);
                if edge < first {
                    second = first;
                    first = edge;
                } else if edge < second {
                    second = edge;
                }
            }

            if required_degree == 1 {
                degree_sum += first;
            } else {
                // A one-customer depot cycle uses the same undirected connection
                // in both directions, so duplicate the only available edge.
                if !second.is_finite() {
                    second = first;
                }
                degree_sum += first + second;
            }
        }

        mst.max(0.5 * degree_sum)
    }

    fn greedy_completion(
        &self,
        current: usize,
        unvisited: &[bool],
        initial_load: f64,
        use_resources: bool,
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut left = remaining.iter().filter(|&&value| value).count();
        let mut position = current;
        let mut load = initial_load.max(0.0);
        let capacity = self.capacity.unwrap_or(f64::MAX);
        let mut cost: f64 = 0.0;

        while left > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for customer in 1..self.node_count {
                if !remaining[customer] {
                    continue;
                }

                if use_resources && load + self.demands[customer] > capacity + 1.0e-9 {
                    continue;
                }

                let edge = self.distance(position, customer);
                if edge < best_cost {
                    best_cost = edge;
                    best_customer = Some(customer);
                }
            }

            if let Some(customer) = best_customer {
                cost += best_cost;
                position = customer;
                if use_resources {
                    load += self.demands[customer];
                }
                remaining[customer] = false;
                left -= 1;
                continue;
            }

            if use_resources && position != 0 {
                // No remaining customer fits. Finish this route and start the
                // next route at the depot, as visit-via-depot does.
                cost += self.distance(position, 0);
                position = 0;
                load = 0.0;
                continue;
            }

            // Defensive fallback for malformed resource data, such as a demand
            // larger than capacity. It also guarantees progress.
            let mut fallback: Option<usize> = None;
            let mut fallback_cost = f64::INFINITY;
            for customer in 1..self.node_count {
                if remaining[customer] {
                    let edge = self.distance(position, customer);
                    if edge < fallback_cost {
                        fallback = Some(customer);
                        fallback_cost = edge;
                    }
                }
            }

            if let Some(customer) = fallback {
                cost += fallback_cost;
                position = customer;
                load = if use_resources {
                    self.demands[customer]
                } else {
                    load
                };
                remaining[customer] = false;
                left -= 1;
            } else {
                break;
            }
        }

        if position != 0 {
            cost += self.distance(position, 0);
        }

        cost
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut depot_name: Option<String> = None;
        let mut customer_names: Vec<String> = Vec::new();
        let mut visited_facts: Vec<ExplicitFact> = Vec::new();

        for goal_index in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_index);
            let fact_name = task.get_fact_name(&fact);

            if let Some(customer) = atom_argument(fact_name, "visited") {
                customer_names.push(customer);
                visited_facts.push(fact.clone());
            } else if let Some(place) = atom_argument(fact_name, "loc") {
                depot_name = Some(place);
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            HeuristicBuildError::from("missing depot location goal fact".to_string())
        })?;

        let mut location_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        for variable_id in 0..task.variables().len() {
            for value in 0..task.variables()[variable_id].domain_size() {
                let fact = ExplicitFact::propositional(variable_id, value);
                if let Some(place) = atom_argument(task.get_fact_name(&fact), "loc") {
                    location_by_name.insert(place, fact);
                }
            }
        }

        let depot_location = location_by_name
            .get(&depot_name)
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                HeuristicBuildError::from("missing depot loc fact".to_string())
            })?;

        let mut location_facts: Vec<ExplicitFact> = Vec::with_capacity(customer_names.len() + 1);
        location_facts.push(depot_location);
        for customer in &customer_names {
            let fact = location_by_name
                .get(customer)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    HeuristicBuildError::from(format!("missing location fact for {}", customer))
                })?;
            location_facts.push(fact);
        }

        let node_count = customer_names.len() + 1;
        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), 0);
        for (index, customer) in customer_names.iter().enumerate() {
            node_by_name.insert(customer.clone(), index + 1);
        }

        let mut distances = vec![f64::INFINITY; node_count * node_count];
        for node in 0..node_count {
            distances[node * node_count + node] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            let (from_name, to_name) = match tokens.as_slice() {
                ["visit", from, to] => (Some(*from), Some(*to)),
                ["return-to-depot", from] => (Some(*from), Some(depot_name.as_str())),
                _ => (None, None),
            };

            let (Some(from_name), Some(to_name)) = (from_name, to_name) else {
                continue;
            };
            let (Some(&from), Some(&to)) = (
                node_by_name.get(from_name),
                node_by_name.get(to_name),
            ) else {
                continue;
            };

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if cost.is_finite() && cost >= 0.0 {
                let cell = from * node_count + to;
                if cost < distances[cell] {
                    distances[cell] = cost;
                }
            }
        }

        // Complete occasional missing arcs by directed shortest paths. On the
        // target CVRP tasks the direct matrix is normally already complete.
        for k in 0..node_count {
            for i in 0..node_count {
                let ik = distances[i * node_count + k];
                if !ik.is_finite() {
                    continue;
                }
                for j in 0..node_count {
                    let kj = distances[k * node_count + j];
                    if !kj.is_finite() {
                        continue;
                    }
                    let candidate = ik + kj;
                    let cell = i * node_count + j;
                    if candidate < distances[cell] {
                        distances[cell] = candidate;
                    }
                }
            }
        }

        let mut largest_finite: f64 = 1.0;
        for &distance in &distances {
            if distance.is_finite() {
                largest_finite = largest_finite.max(distance);
            }
        }
        let missing_arc_cost = largest_finite * (node_count.max(2) as f64);
        for distance in &mut distances {
            if !distance.is_finite() {
                *distance = missing_arc_cost;
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let load_id = exact_numeric_id(task, "load()");
        let vehicles_id = exact_numeric_id(task, "vehicles()");
        let capacity_id = exact_numeric_id(task, "capacity()");
        let max_vehicles_id = exact_numeric_id(task, "max_vehicles()");

        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0);
        let max_vehicles = max_vehicles_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0);

        let mut demands = vec![0.0; node_count];
        let mut all_demands_found = true;
        for (index, customer) in customer_names.iter().enumerate() {
            let exact_name = format!("demand({})", customer);
            if let Some(id) = exact_numeric_id(task, &exact_name) {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value >= 0.0 {
                        demands[index + 1] = *value;
                        continue;
                    }
                }
            }
            all_demands_found = false;
        }

        let resource_information_complete =
            load_id.is_some() && capacity.is_some() && all_demands_found;
        let dead_end_penalty = missing_arc_cost * (node_count.max(2) as f64) * 100.0;

        Ok(Self {
            visited_facts,
            location_facts,
            distances,
            node_count,
            numeric_count: task.numeric_variables().len(),
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            resource_information_complete,
            dead_end_penalty,
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

        let mut current = 0;
        for (node, fact) in self.location_facts.iter().enumerate() {
            if fact.is_hold(view) {
                current = node;
                break;
            }
        }

        let mut unvisited = vec![false; self.node_count];
        let mut unvisited_count = 0usize;
        for (customer_index, fact) in self.visited_facts.iter().enumerate() {
            if !fact.is_hold(view) {
                unvisited[customer_index + 1] = true;
                unvisited_count += 1;
            }
        }

        if unvisited_count == 0 {
            return Ok(self.distance(current, 0).max(0.0));
        }

        let mut load: f64 = 0.0;
        let mut vehicles: Option<f64> = None;

        if self.resource_information_complete
            || (self.vehicles_id.is_some() && self.max_vehicles.is_some())
        {
            let mut numeric_values = vec![0.0; self.numeric_count];
            view.fill_numeric(&mut numeric_values).map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

            if let Some(id) = self.load_id {
                load = numeric_values.get(id).copied().unwrap_or(0.0).max(0.0);
            }
            if let Some(id) = self.vehicles_id {
                vehicles = numeric_values.get(id).copied();
            }
        }

        if self.resource_information_complete {
            if let (Some(capacity), Some(max_vehicles), Some(used_vehicles)) =
                (self.capacity, self.max_vehicles, vehicles)
            {
                let mut remaining_demand: f64 = 0.0;
                for customer in 1..self.node_count {
                    if unvisited[customer] {
                        remaining_demand += self.demands[customer];
                    }
                }

                let current_free_capacity = (capacity - load).max(0.0);
                let excess = (remaining_demand - current_free_capacity).max(0.0);
                let additional_routes = if excess <= 1.0e-9 {
                    0.0
                } else {
                    (excess / capacity - 1.0e-12).ceil()
                };

                if used_vehicles + additional_routes > max_vehicles + 1.0e-9 {
                    return Ok(self.dead_end_penalty);
                }
            }
        }

        let lower_bound = self.routing_lower_bound(current, &unvisited);
        let greedy = self.greedy_completion(
            current,
            &unvisited,
            load,
            self.resource_information_complete,
        );

        // The structural relaxation prevents severe underestimation, while a
        // slightly discounted greedy completion supplies useful non-admissible
        // route-ordering and capacity guidance.
        let h = lower_bound.max(0.90 * greedy).max(0.0);
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
