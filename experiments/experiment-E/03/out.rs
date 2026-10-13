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

struct CustomerInfo {
    visited_fact: ExplicitFact,
    demand: Option<f64>,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    location_facts: Vec<ExplicitFact>,
    distances: Vec<f64>,
    node_count: usize,
    load_id: Option<usize>,
    numeric_count: usize,
    capacity: Option<f64>,
    missing_edge_cost: f64,
}

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let text = name.trim();
    let text = text.strip_prefix("Atom ").unwrap_or(text);
    let open = text.find('(')?;
    let close = text.rfind(')')?;
    if close <= open || text[..open].trim() != predicate {
        return None;
    }

    let argument = text[open + 1..close].trim();
    if argument.is_empty() || argument.contains(',') {
        return None;
    }
    Some(argument.to_string())
}

fn normalized_numeric_name(name: &str) -> &str {
    let text = name.trim();
    text.strip_prefix("PNE ").unwrap_or(text).trim()
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut depot_name: Option<String> = None;
        let mut customer_names: Vec<String> = Vec::new();

        for goal_id in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_id);
            let fact_name = task.get_fact_name(&fact);

            if let Some(place) = atom_argument(fact_name, "loc") {
                depot_name = Some(place);
            }
            if let Some(customer) = atom_argument(fact_name, "visited") {
                customer_names.push(customer);
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "missing customer visited goal facts".to_string(),
            ));
        }

        customer_names.sort();
        customer_names.dedup();

        let mut visited_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        let mut location_by_name: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = atom_argument(fact_name, "visited") {
                    visited_by_name.insert(customer, fact.clone());
                }
                if let Some(place) = atom_argument(fact_name, "loc") {
                    location_by_name.insert(place, fact);
                }
            }
        }

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name.clone());
        node_names.extend(customer_names.iter().cloned());

        let mut location_facts = Vec::with_capacity(node_names.len());
        for node_name in &node_names {
            let fact = location_by_name
                .get(node_name)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("missing location fact for {node_name}").into()
                })?;
            location_facts.push(fact);
        }

        let numeric_variables = task.numeric_variables();
        let numeric_count = numeric_variables.len();
        let initial_numeric = task.get_initial_numeric_state_values();

        let mut load_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut demand_by_name: HashMap<String, f64> = HashMap::new();

        for (id, variable) in numeric_variables.iter().enumerate() {
            let name = normalized_numeric_name(variable.name());
            if name == "load()" {
                load_id = Some(id);
            } else if name == "capacity()" {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value > 0.0 {
                        capacity = Some(*value);
                    }
                }
            }
        }

        // Demand fluents may have been folded away by the translator. Match only
        // complete exact names; synthesized derived expressions are never used.
        for customer_name in &customer_names {
            let expected = format!("demand({customer_name})");
            for (id, variable) in numeric_variables.iter().enumerate() {
                if normalized_numeric_name(variable.name()) == expected {
                    if let Some(value) = initial_numeric.get(id) {
                        if value.is_finite() && *value >= 0.0 {
                            demand_by_name.insert(customer_name.clone(), *value);
                        }
                    }
                    break;
                }
            }
        }

        let mut customers = Vec::with_capacity(customer_names.len());
        for customer_name in &customer_names {
            let visited_fact = visited_by_name
                .get(customer_name)
                .cloned()
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("missing visited fact for {customer_name}").into()
                })?;
            customers.push(CustomerInfo {
                visited_fact,
                demand: demand_by_name.get(customer_name).copied(),
            });
        }

        let node_count = node_names.len();
        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let mut distances = vec![f64::INFINITY; node_count * node_count];
        for i in 0..node_count {
            distances[i * node_count + i] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens: Vec<&str> = operator.name().split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            let endpoints = match tokens.as_slice() {
                ["visit", from, to] => Some((*from, *to)),
                ["return-to-depot", from] => Some((*from, depot_name.as_str())),
                _ => None,
            };

            if let Some((from, to)) = endpoints {
                if let (Some(&from_id), Some(&to_id)) =
                    (node_index.get(from), node_index.get(to))
                {
                    let entry = &mut distances[from_id * node_count + to_id];
                    if cost < *entry {
                        *entry = cost;
                    }
                }
            }
        }

        let mut largest_edge: f64 = 1.0;
        for value in &distances {
            if value.is_finite() {
                largest_edge = largest_edge.max(*value);
            }
        }
        let missing_edge_cost = largest_edge * (node_count as f64 + 1.0);

        Ok(Self {
            customers,
            location_facts,
            distances,
            node_count,
            load_id,
            numeric_count,
            capacity,
            missing_edge_cost,
        })
    }

    fn directed_distance(&self, from: usize, to: usize) -> f64 {
        let direct = self.distances[from * self.node_count + to];
        if direct.is_finite() {
            return direct;
        }

        let reverse = self.distances[to * self.node_count + from];
        if reverse.is_finite() {
            reverse
        } else {
            self.missing_edge_cost
        }
    }

    fn undirected_distance(&self, a: usize, b: usize) -> f64 {
        let ab = self.distances[a * self.node_count + b];
        let ba = self.distances[b * self.node_count + a];
        match (ab.is_finite(), ba.is_finite()) {
            (true, true) => ab.min(ba),
            (true, false) => ab,
            (false, true) => ba,
            (false, false) => self.missing_edge_cost,
        }
    }

    fn mst_bound(&self, relevant: &[bool]) -> f64 {
        let relevant_count = relevant.iter().filter(|included| **included).count();
        if relevant_count <= 1 {
            return 0.0;
        }

        let mut in_tree = vec![false; self.node_count];
        let mut key = vec![f64::INFINITY; self.node_count];
        let start = relevant.iter().position(|included| *included).unwrap_or(0);
        key[start] = 0.0;

        let mut total: f64 = 0.0;
        for _ in 0..relevant_count {
            let mut chosen: Option<usize> = None;
            let mut chosen_key = f64::INFINITY;
            for node in 0..self.node_count {
                if relevant[node] && !in_tree[node] && key[node] < chosen_key {
                    chosen = Some(node);
                    chosen_key = key[node];
                }
            }

            let Some(node) = chosen else {
                return total + self.missing_edge_cost;
            };
            in_tree[node] = true;
            total += chosen_key;

            for other in 0..self.node_count {
                if relevant[other] && !in_tree[other] {
                    let edge = self.undirected_distance(node, other);
                    if edge < key[other] {
                        key[other] = edge;
                    }
                }
            }
        }
        total
    }

    fn degree_bound(
        &self,
        relevant: &[bool],
        current: usize,
        unvisited: &[bool],
    ) -> f64 {
        let relevant_count = relevant.iter().filter(|included| **included).count();
        if relevant_count <= 1 {
            return 0.0;
        }

        let mut degree_sum: f64 = 0.0;
        for node in 0..self.node_count {
            if !relevant[node] {
                continue;
            }

            let required_degree = if node > 0 && unvisited[node - 1] {
                2
            } else if current == 0 && node == 0 {
                2
            } else if node == current || node == 0 {
                1
            } else {
                0
            };

            if required_degree == 0 {
                continue;
            }

            let mut best = f64::INFINITY;
            let mut second = f64::INFINITY;
            for other in 0..self.node_count {
                if other == node || !relevant[other] {
                    continue;
                }
                let edge = self.undirected_distance(node, other);
                if edge < best {
                    second = best;
                    best = edge;
                } else if edge < second {
                    second = edge;
                }
            }

            if best.is_finite() {
                degree_sum += best;
            }
            if required_degree == 2 {
                degree_sum += if second.is_finite() { second } else { best };
            }
        }

        0.5 * degree_sum
    }

    fn radial_capacity_bound(&self, unvisited: &[bool]) -> f64 {
        let Some(capacity) = self.capacity else {
            return 0.0;
        };
        if capacity <= 0.0 {
            return 0.0;
        }

        let mut weighted_round_trips: f64 = 0.0;
        for (customer_index, customer) in self.customers.iter().enumerate() {
            if !unvisited[customer_index] {
                continue;
            }
            let Some(demand) = customer.demand else {
                return 0.0;
            };
            let node = customer_index + 1;
            let round_trip = self.directed_distance(0, node)
                + self.directed_distance(node, 0);
            weighted_round_trips += demand * round_trip;
        }

        weighted_round_trips / capacity
    }

    fn greedy_completion(
        &self,
        current: usize,
        unvisited: &[bool],
        current_load: Option<f64>,
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|value| **value).count();
        let mut position = current;
        let mut load = current_load.unwrap_or(0.0).max(0.0);

        let resource_known = self.capacity.is_some()
            && current_load.is_some()
            && self.customers.iter().enumerate().all(|(i, customer)| {
                !remaining[i] || customer.demand.is_some()
            });

        let mut total: f64 = 0.0;
        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for customer_index in 0..self.customers.len() {
                if !remaining[customer_index] {
                    continue;
                }

                if resource_known {
                    let demand = self.customers[customer_index].demand.unwrap_or(0.0);
                    let capacity = self.capacity.unwrap_or(f64::MAX);
                    if load + demand > capacity + 1.0e-9 {
                        continue;
                    }
                }

                let node = customer_index + 1;
                let cost = self.directed_distance(position, node);
                if cost < best_cost {
                    best_cost = cost;
                    best_customer = Some(customer_index);
                }
            }

            if let Some(customer_index) = best_customer {
                total += best_cost;
                position = customer_index + 1;
                if resource_known {
                    load += self.customers[customer_index].demand.unwrap_or(0.0);
                }
                remaining[customer_index] = false;
                remaining_count -= 1;
                continue;
            }

            // No remaining customer fits the current vehicle. Model a
            // visit-via-depot transition and start the next route.
            let mut restart_customer: Option<usize> = None;
            let mut restart_cost = f64::INFINITY;
            for customer_index in 0..self.customers.len() {
                if !remaining[customer_index] {
                    continue;
                }
                let node = customer_index + 1;
                let cost = self.directed_distance(position, 0)
                    + self.directed_distance(0, node);
                if cost < restart_cost {
                    restart_cost = cost;
                    restart_customer = Some(customer_index);
                }
            }

            let Some(customer_index) = restart_customer else {
                break;
            };
            total += restart_cost;
            position = customer_index + 1;
            load = self.customers[customer_index].demand.unwrap_or(0.0);
            remaining[customer_index] = false;
            remaining_count -= 1;
        }

        total + self.directed_distance(position, 0)
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

        let current = self.location_facts
            .iter()
            .position(|fact| fact.is_hold(view))
            .unwrap_or(0);

        let mut unvisited = vec![false; self.customers.len()];
        let mut unvisited_count = 0usize;
        for (index, customer) in self.customers.iter().enumerate() {
            if !customer.visited_fact.is_hold(view) {
                unvisited[index] = true;
                unvisited_count += 1;
            }
        }

        if unvisited_count == 0 {
            return Ok(self.directed_distance(current, 0).max(0.0));
        }

        let mut current_load: Option<f64> = None;
        if let Some(load_id) = self.load_id {
            let mut numeric_values = vec![0.0; self.numeric_count];
            view.fill_numeric(&mut numeric_values).map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;
            if let Some(value) = numeric_values.get(load_id) {
                if value.is_finite() {
                    current_load = Some(*value);
                }
            }
        }

        let mut relevant = vec![false; self.node_count];
        relevant[0] = true;
        relevant[current] = true;
        for customer_index in 0..self.customers.len() {
            if unvisited[customer_index] {
                relevant[customer_index + 1] = true;
            }
        }

        let mst = self.mst_bound(&relevant);
        let degree = self.degree_bound(&relevant, current, &unvisited);
        let radial = self.radial_capacity_bound(&unvisited);
        let lower_estimate = mst.max(degree).max(radial);

        let greedy = self.greedy_completion(current, &unvisited, current_load);

        // The graph bounds provide stable global structure, while the greedy
        // completion captures route ordering and forced capacity resets. The
        // larger weight on the completion estimate gives useful discrimination
        // between successor states without fully trusting a myopic rollout.
        let blended = 0.70 * greedy + 0.30 * lower_estimate;
        let h = blended.max(lower_estimate).max(0.0);

        if h.is_finite() {
            Ok(h)
        } else {
            Ok(self.missing_edge_cost.max(0.0))
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
