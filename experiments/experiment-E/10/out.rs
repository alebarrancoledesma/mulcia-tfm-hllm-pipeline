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

fn parse_unary_atom(name: &str, predicate: &str) -> Option<String> {
    let text = name.trim();
    let text = text.strip_prefix("Atom ").unwrap_or(text).trim();
    let open = text.find('(')?;
    let close = text.rfind(')')?;

    if text[..open].trim() != predicate {
        return None;
    }

    let argument = text[open + 1..close].trim();
    if argument.is_empty() || argument.contains(',') || argument.split_whitespace().count() != 1 {
        return None;
    }

    Some(argument.to_string())
}

fn operator_tokens(name: &str) -> Vec<String> {
    name.replace('(', " ")
        .replace(')', " ")
        .replace(',', " ")
        .split_whitespace()
        .map(|part| part.to_string())
        .collect()
}

fn object_index(objects: &[String], name: &str) -> Option<usize> {
    objects.iter().position(|object| object == name)
}

struct LlmHeuristic {
    // Node zero is the depot; nodes 1.. are customers.
    objects: Vec<String>,
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,

    // Cheapest routing action from one represented node to another. For a
    // customer-to-customer transition this is the minimum of a direct visit
    // and a visit-via-depot action.
    edge_costs: Vec<f64>,
    node_count: usize,
    fallback_cost: f64,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        // Customer identities and their visited facts are defined by the
        // goals, rather than by hardcoded object names or variable IDs.
        for goal_index in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_index);
            let fact_name = task.get_fact_name(&fact);

            if let Some(customer) = parse_unary_atom(fact_name, "visited") {
                if !customer_names.iter().any(|known| known == &customer) {
                    customer_names.push(customer);
                }
            } else if let Some(place) = parse_unary_atom(fact_name, "loc") {
                depot_name = Some(place);
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "no CVRP customer goals were found".to_string(),
            ));
        }

        let mut objects = Vec::with_capacity(customer_names.len() + 1);
        objects.push(depot_name);
        objects.extend(customer_names.iter().cloned());

        let node_count = objects.len();
        let mut location_facts: Vec<Option<ExplicitFact>> =
            (0..node_count).map(|_| None).collect();
        let mut visited_facts: Vec<Option<ExplicitFact>> =
            (0..customer_names.len()).map(|_| None).collect();

        // Resolve propositional facts once. Truth is never inferred from a
        // particular SAS value such as zero.
        for variable_id in 0..task.variables().len() {
            let domain_size = task.variables()[variable_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(place) = parse_unary_atom(fact_name, "loc") {
                    if let Some(index) = object_index(&objects, &place) {
                        location_facts[index] = Some(fact);
                    }
                } else if let Some(customer) = parse_unary_atom(fact_name, "visited") {
                    if let Some(index) = object_index(&customer_names, &customer) {
                        visited_facts[index] = Some(fact);
                    }
                }
            }
        }

        let location_facts: Vec<ExplicitFact> = location_facts
            .into_iter()
            .enumerate()
            .map(|(index, fact)| {
                fact.ok_or_else(|| -> HeuristicBuildError {
                    format!("missing location fact for routing node {index}").into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let visited_facts: Vec<ExplicitFact> = visited_facts
            .into_iter()
            .enumerate()
            .map(|(index, fact)| {
                fact.ok_or_else(|| -> HeuristicBuildError {
                    format!("missing visited fact for customer index {index}").into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let matrix_size = node_count * node_count;
        let mut direct = vec![f64::INFINITY; matrix_size];
        let mut via_depot = vec![f64::INFINITY; matrix_size];
        for index in 0..node_count {
            direct[index * node_count + index] = 0.0;
        }

        // Grounded action names provide the object mapping, while the metric
        // helper provides the actual compiled operator cost.
        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0].as_str() {
                "visit" if tokens.len() == 3 => {
                    if let (Some(from), Some(to)) = (
                        object_index(&objects, &tokens[1]),
                        object_index(&objects, &tokens[2]),
                    ) {
                        let slot = from * node_count + to;
                        direct[slot] = direct[slot].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(from), Some(to)) = (
                        object_index(&objects, &tokens[1]),
                        object_index(&objects, &tokens[2]),
                    ) {
                        let slot = from * node_count + to;
                        via_depot[slot] = via_depot[slot].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(from) = object_index(&objects, &tokens[1]) {
                        direct[from * node_count] = direct[from * node_count].min(cost);
                    }
                }
                _ => {}
            }
        }

        let mut edge_costs = direct;
        for from in 1..node_count {
            for to in 1..node_count {
                if from != to {
                    let slot = from * node_count + to;
                    edge_costs[slot] = edge_costs[slot].min(via_depot[slot]);
                }
            }
        }

        let mut maximum_finite_edge: f64 = 1.0;
        for &cost in &edge_costs {
            if cost.is_finite() {
                maximum_finite_edge = maximum_finite_edge.max(cost);
            }
        }
        let fallback_cost = maximum_finite_edge * (node_count as f64 + 1.0);

        Ok(Self {
            objects,
            visited_facts,
            location_facts,
            edge_costs,
            node_count,
            fallback_cost,
        })
    }

    #[inline]
    fn edge(&self, from: usize, to: usize) -> f64 {
        self.edge_costs[from * self.node_count + to]
    }

    fn routing_lower_bound(&self, current: usize, unvisited: &[usize]) -> f64 {
        if unvisited.is_empty() {
            let return_cost = self.edge(current, 0);
            return if return_cost.is_finite() {
                return_cost
            } else {
                self.fallback_cost
            };
        }

        let count = unvisited.len();

        // Prim MST over remaining customers using the cheaper orientation.
        let mut in_tree = vec![false; count];
        let mut keys = vec![f64::INFINITY; count];
        keys[0] = 0.0;
        let mut mst_cost: f64 = 0.0;

        for _ in 0..count {
            let mut best_position: Option<usize> = None;
            let mut best_key = f64::INFINITY;
            for position in 0..count {
                if !in_tree[position] && keys[position] < best_key {
                    best_key = keys[position];
                    best_position = Some(position);
                }
            }

            let Some(position) = best_position else {
                return self.fallback_cost;
            };
            if !best_key.is_finite() {
                return self.fallback_cost;
            }

            in_tree[position] = true;
            mst_cost += best_key;
            let from = unvisited[position];

            for other_position in 0..count {
                if !in_tree[other_position] {
                    let to = unvisited[other_position];
                    let undirected = self.edge(from, to).min(self.edge(to, from));
                    if undirected < keys[other_position] {
                        keys[other_position] = undirected;
                    }
                }
            }
        }

        let mut start_connector = f64::INFINITY;
        let mut depot_connector = f64::INFINITY;
        for &customer in unvisited {
            start_connector = start_connector.min(self.edge(current, customer));
            depot_connector = depot_connector.min(self.edge(customer, 0));
        }

        let tree_bound = mst_cost + start_connector + depot_connector;

        // Directed outgoing-degree relaxation: the current position must enter
        // the remaining set, and each customer must leave toward another
        // remaining customer or the depot.
        let mut degree_bound = start_connector;
        for &from in unvisited {
            let mut cheapest_exit = self.edge(from, 0);
            for &to in unvisited {
                if from != to {
                    cheapest_exit = cheapest_exit.min(self.edge(from, to));
                }
            }
            degree_bound += cheapest_exit;
        }

        let bound = tree_bound.max(degree_bound);
        if bound.is_finite() {
            bound
        } else {
            self.fallback_cost
        }
    }

    fn greedy_completion(&self, current: usize, unvisited: &[usize]) -> f64 {
        if unvisited.is_empty() {
            let cost = self.edge(current, 0);
            return if cost.is_finite() { cost } else { self.fallback_cost };
        }

        let mut selected = vec![false; unvisited.len()];
        let mut position = current;
        let mut total: f64 = 0.0;

        for _ in 0..unvisited.len() {
            let mut best: Option<(usize, usize, f64)> = None;

            for (candidate_position, &customer) in unvisited.iter().enumerate() {
                if selected[candidate_position] {
                    continue;
                }

                let cost = self.edge(position, customer);
                if best.map_or(true, |(_, _, best_cost)| cost < best_cost) {
                    best = Some((candidate_position, customer, cost));
                }
            }

            let Some((chosen_position, customer, cost)) = best else {
                return self.fallback_cost;
            };
            if !cost.is_finite() {
                return self.fallback_cost;
            }

            selected[chosen_position] = true;
            total += cost;
            position = customer;
        }

        let return_cost = self.edge(position, 0);
        if !return_cost.is_finite() {
            return self.fallback_cost;
        }

        total + return_cost
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

        let mut current_location: Option<usize> = None;
        for (node, fact) in self.location_facts.iter().enumerate() {
            if fact.is_hold(registry.view(state)) {
                current_location = Some(node);
                break;
            }
        }

        let current_location = current_location.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "CVRP state has no recognized current location".to_string(),
            )
        })?;

        let mut unvisited: Vec<usize> = Vec::with_capacity(self.objects.len() - 1);
        for (customer_index, fact) in self.visited_facts.iter().enumerate() {
            if !fact.is_hold(registry.view(state)) {
                // Customer i corresponds to routing node i + 1.
                unvisited.push(customer_index + 1);
            }
        }

        if unvisited.is_empty() {
            let return_cost = self.edge(current_location, 0);
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                self.fallback_cost
            });
        }

        let lower = self.routing_lower_bound(current_location, &unvisited);
        let greedy = self.greedy_completion(current_location, &unvisited);

        // The lower-bound component is stable and globally informed; the
        // greedy component supplies stronger route-ordering information. The
        // heuristic is deliberately not required to be admissible.
        let h = (0.65 * lower + 0.35 * greedy).max(lower).max(0.0);
        Ok(if h.is_finite() { h } else { self.fallback_cost })
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
