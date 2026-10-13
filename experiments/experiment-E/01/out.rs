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

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let trimmed = name.trim();
    let atom = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);
    let rest = atom.strip_prefix(&prefix)?;
    let argument = rest.strip_suffix(')')?;
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.trim().to_string())
    }
}

fn operator_tokens(name: &str) -> Vec<String> {
    name.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c| c == '(' || c == ')')
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn exact_numeric_id(
    task: &dyn AbstractNumericTask,
    plain_name: &str,
) -> Option<usize> {
    let prefixed_name = format!("PNE {}", plain_name);
    task.numeric_variables()
        .iter()
        .enumerate()
        .find_map(|(id, variable)| {
            let name = variable.name().trim();
            if name == plain_name || name == prefixed_name {
                Some(id)
            } else {
                None
            }
        })
}

struct LlmHeuristic {
    // Node zero is the depot; node i + 1 is customer i.
    location_facts: Vec<ExplicitFact>,
    visited_facts: Vec<ExplicitFact>,
    distances: Vec<Vec<f64>>,

    // Resource information is deliberately optional because the translator may
    // fold capacity and demand expressions into anonymous derived variables.
    load_id: Option<usize>,
    capacity: Option<f64>,
    demands: Option<Vec<f64>>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut depot_name: Option<String> = None;
        let mut depot_goal_fact: Option<ExplicitFact> = None;
        let mut customer_names: Vec<String> = Vec::new();
        let mut visited_facts: Vec<ExplicitFact> = Vec::new();

        // In this domain the goal identifies both the depot location and the
        // complete set of customers that must be visited.
        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            let goal_name = task.get_fact_name(goal);

            if let Some(customer) = atom_argument(goal_name, "visited") {
                customer_names.push(customer);
                visited_facts.push(goal.clone());
            } else if let Some(place) = atom_argument(goal_name, "loc") {
                depot_name = Some(place);
                depot_goal_fact = Some(goal.clone());
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;
        let depot_goal_fact = depot_goal_fact.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP task has no visited-customer goals".to_string(),
            ));
        }

        // Resolve every customer's location fact once. Propositional variable
        // names are generic, so semantic fact names must be inspected.
        let mut customer_location_facts: Vec<Option<ExplicitFact>> =
            vec![None; customer_names.len()];

        for variable_id in 0..task.variables().len() {
            let variable = &task.variables()[variable_id];
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(variable_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(place) = atom_argument(fact_name, "loc") {
                    if let Some(customer_index) =
                        customer_names.iter().position(|name| name == &place)
                    {
                        customer_location_facts[customer_index] = Some(fact);
                    }
                }
            }
        }

        let mut location_facts = Vec::with_capacity(customer_names.len() + 1);
        location_facts.push(depot_goal_fact);
        for (customer_index, fact) in customer_location_facts.into_iter().enumerate() {
            let fact = fact.ok_or_else(|| -> HeuristicBuildError {
                format!(
                    "missing location fact for customer {}",
                    customer_names[customer_index]
                )
                .into()
            })?;
            location_facts.push(fact);
        }

        let node_count = customer_names.len() + 1;
        let mut node_names = Vec::with_capacity(node_count);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());

        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for node in 0..node_count {
            distances[node][node] = 0.0;
        }

        // Reconstruct the directed routing matrix from grounded operators.
        // This avoids all dependence on folded travel-cost numeric variables.
        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            if tokens[0] == "visit" && tokens.len() == 3 {
                let from = node_names.iter().position(|name| name == &tokens[1]);
                let to = node_names.iter().position(|name| name == &tokens[2]);
                if let (Some(from), Some(to)) = (from, to) {
                    distances[from][to] = distances[from][to].min(cost);
                }
            } else if tokens[0] == "return-to-depot" && tokens.len() == 2 {
                if let Some(from) = node_names.iter().position(|name| name == &tokens[1]) {
                    distances[from][0] = distances[from][0].min(cost);
                }
            }
        }

        // The CVRP domain supplies a complete customer graph. Failing loudly
        // here is preferable to producing infinities during millions of calls.
        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !distances[from][to].is_finite() {
                    return Err(HeuristicBuildError::from(
                        format!(
                            "missing routing cost from {} to {}",
                            node_names[from], node_names[to]
                        ),
                    ));
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let load_id = exact_numeric_id(task, "load()");

        let capacity = exact_numeric_id(task, "capacity()").and_then(|id| {
            initial_numeric
                .get(id)
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
        });

        let mut recovered_demands = Vec::with_capacity(customer_names.len());
        let mut all_demands_available = true;
        for customer in &customer_names {
            let demand_name = format!("demand({})", customer);
            let demand = exact_numeric_id(task, &demand_name).and_then(|id| {
                initial_numeric
                    .get(id)
                    .copied()
                    .filter(|value| value.is_finite() && *value >= 0.0)
            });
            if let Some(value) = demand {
                recovered_demands.push(value);
            } else {
                all_demands_available = false;
                break;
            }
        }

        let demands = if all_demands_available {
            Some(recovered_demands)
        } else {
            None
        };

        Ok(Self {
            location_facts,
            visited_facts,
            distances,
            load_id,
            capacity,
            demands,
        })
    }

    fn undirected_distance(&self, a: usize, b: usize) -> f64 {
        self.distances[a][b].min(self.distances[b][a])
    }

    // Prim's algorithm on the depot, current position, and unvisited customers.
    fn mst_cost(&self, nodes: &[usize]) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let mut in_tree = vec![false; nodes.len()];
        let mut best = vec![f64::INFINITY; nodes.len()];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..nodes.len() {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;
            for index in 0..nodes.len() {
                if !in_tree[index] && best[index] < selected_cost {
                    selected = Some(index);
                    selected_cost = best[index];
                }
            }

            let Some(selected) = selected else {
                break;
            };
            in_tree[selected] = true;
            total += selected_cost;

            for other in 0..nodes.len() {
                if !in_tree[other] {
                    let edge = self.undirected_distance(nodes[selected], nodes[other]);
                    if edge < best[other] {
                        best[other] = edge;
                    }
                }
            }
        }

        total
    }

    // Simulates a cheap nearest-neighbour CVRP completion. If exact capacity,
    // demands, and current load survived translation, route resets are inserted
    // whenever the next customer does not fit. Otherwise it degrades to a TSP
    // nearest-neighbour completion while retaining the full routing signal.
    fn greedy_completion(
        &self,
        current_node: usize,
        unvisited: &[usize],
        current_load: Option<f64>,
    ) -> f64 {
        let resource_data = match (
            self.capacity,
            self.demands.as_ref(),
            current_load,
        ) {
            (Some(capacity), Some(demands), Some(load)) => {
                Some((capacity, demands, load.max(0.0)))
            }
            _ => None,
        };

        let mut remaining = unvisited.to_vec();
        let mut node = current_node;
        let mut load = resource_data.map(|(_, _, value)| value).unwrap_or(0.0);
        let mut total: f64 = 0.0;

        while !remaining.is_empty() {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;

            for (position, &candidate) in remaining.iter().enumerate() {
                let feasible = if let Some((capacity, demands, _)) = resource_data {
                    let demand = demands[candidate - 1];
                    load + demand <= capacity + 1.0e-9
                } else {
                    true
                };

                let edge = self.distances[node][candidate];
                if feasible && edge < selected_cost {
                    selected = Some(position);
                    selected_cost = edge;
                }
            }

            if selected.is_none() && resource_data.is_some() && node != 0 {
                total += self.distances[node][0];
                node = 0;
                load = 0.0;
                continue;
            }

            // An oversized demand should be impossible in valid instances, but
            // ignoring capacity here avoids an infinite loop on malformed data.
            if selected.is_none() {
                for (position, &candidate) in remaining.iter().enumerate() {
                    let edge = self.distances[node][candidate];
                    if edge < selected_cost {
                        selected = Some(position);
                        selected_cost = edge;
                    }
                }
            }

            let Some(position) = selected else {
                break;
            };
            let customer_node = remaining.swap_remove(position);
            total += self.distances[node][customer_node];
            node = customer_node;

            if let Some((_, demands, _)) = resource_data {
                load += demands[customer_node - 1];
            }
        }

        if node != 0 {
            total += self.distances[node][0];
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

        let current_node = self
            .location_facts
            .iter()
            .position(|fact| fact.is_hold(registry.view(state)))
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "CVRP state has no recognized current location".to_string(),
                )
            })?;

        let mut unvisited = Vec::with_capacity(self.visited_facts.len());
        for (customer_index, visited_fact) in self.visited_facts.iter().enumerate() {
            if !visited_fact.is_hold(registry.view(state)) {
                unvisited.push(customer_index + 1);
            }
        }

        if unvisited.is_empty() {
            return Ok(self.distances[current_node][0].max(0.0));
        }

        let resource_information_available =
            self.capacity.is_some() && self.demands.is_some() && self.load_id.is_some();

        let current_load = if resource_information_available {
            let mut numeric_values = vec![0.0; task.numeric_variables().len()];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
            self.load_id.and_then(|id| numeric_values.get(id).copied())
        } else {
            None
        };

        let greedy = self.greedy_completion(current_node, &unvisited, current_load);

        let mut mst_nodes = Vec::with_capacity(unvisited.len() + 2);
        mst_nodes.push(0);
        if current_node != 0 {
            mst_nodes.push(current_node);
        }
        mst_nodes.extend(unvisited.iter().copied());
        let mst = self.mst_cost(&mst_nodes);

        // A classical radial CVRP signal estimates the depot travel forced by
        // demand. Subtracting the current depot radius partially accounts for
        // the already-started route represented by the current state.
        let mut structural_bound = mst;
        if let (Some(capacity), Some(demands)) = (self.capacity, self.demands.as_ref()) {
            let mut weighted_radius: f64 = 0.0;
            for &customer_node in &unvisited {
                weighted_radius +=
                    demands[customer_node - 1] * self.distances[0][customer_node];
            }
            let current_radius = if current_node == 0 {
                0.0
            } else {
                self.distances[0][current_node]
            };
            let radial = (2.0 * weighted_radius / capacity - current_radius).max(0.0);
            structural_bound = structural_bound.max(radial);
        }

        // The rollout is informative about route ordering, while the MST/radial
        // component is smoother and less sensitive to a bad greedy choice.
        let h = (0.75 * greedy + 0.25 * structural_bound).max(0.0);
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
