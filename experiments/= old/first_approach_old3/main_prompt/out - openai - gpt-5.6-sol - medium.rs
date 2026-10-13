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
struct CustomerInfo {
    visited_fact: ExplicitFact,
    loc_fact: Option<ExplicitFact>,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    depot_loc_fact: Option<ExplicitFact>,
    directed_cost: Vec<f64>,
    symmetric_cost: Vec<f64>,
    node_count: usize,
}

fn atom_argument<'a>(fact_name: &'a str, predicate: &str) -> Option<&'a str> {
    let prefix = format!("Atom {}(", predicate);
    fact_name
        .strip_prefix(&prefix)
        .and_then(|rest| rest.strip_suffix(')'))
}

fn clean_operator_name(name: &str) -> &str {
    name.trim().trim_start_matches('(').trim_end_matches(')')
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover the positive visited and location facts from their semantic
        // fact names. Propositional variable/value IDs are never assumed.
        let mut visited_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        let mut loc_by_name: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = atom_argument(fact_name, "visited") {
                    visited_by_name.insert(object.to_string(), fact.clone());
                }
                if let Some(object) = atom_argument(fact_name, "loc") {
                    loc_by_name.insert(object.to_string(), fact.clone());
                }
            }
        }

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        // In this domain the unique place with a loc fact but no visited fact is
        // the depot. Prefer one that is explicitly a location goal.
        let mut goal_depot_name: Option<String> = None;
        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
            let goal_name = task.get_fact_name(&goal);
            if let Some(object) = atom_argument(goal_name, "loc") {
                if !visited_by_name.contains_key(object) {
                    goal_depot_name = Some(object.to_string());
                    break;
                }
            }
        }

        let depot_name = goal_depot_name.or_else(|| {
            let mut candidates: Vec<String> = loc_by_name
                .keys()
                .filter(|name| !visited_by_name.contains_key(*name))
                .cloned()
                .collect();
            candidates.sort();
            candidates.into_iter().next()
        });

        let mut customers = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            if let Some(visited_fact) = visited_by_name.get(name) {
                customers.push(CustomerInfo {
                    visited_fact: visited_fact.clone(),
                    loc_fact: loc_by_name.get(name).cloned(),
                });
            }
        }

        // Node 0 is the depot; customer i is node i+1.
        let node_count = customers.len() + 1;
        let mut object_to_node: HashMap<String, usize> = HashMap::new();
        if let Some(name) = &depot_name {
            object_to_node.insert(name.clone(), 0);
        }
        for (customer_id, name) in customer_names.iter().enumerate() {
            object_to_node.insert(name.clone(), customer_id + 1);
        }

        let mut directed_cost = vec![f64::INFINITY; node_count * node_count];
        for node in 0..node_count {
            directed_cost[node * node_count + node] = 0.0;
        }

        // Reconstruct routing costs from grounded operators. This remains valid
        // when travel-cost fluents have been compiled into derived expressions.
        for op in task.get_operators() {
            let clean_name = clean_operator_name(op.name());
            let parts: Vec<&str> = clean_name.split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, op);

            if parts.len() == 3 && parts[0] == "visit" {
                if let (Some(&from), Some(&to)) = (
                    object_to_node.get(parts[1]),
                    object_to_node.get(parts[2]),
                ) {
                    let slot = from * node_count + to;
                    if cost.is_finite() && cost >= 0.0 && cost < directed_cost[slot] {
                        directed_cost[slot] = cost;
                    }
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&from) = object_to_node.get(parts[1]) {
                    let slot = from * node_count;
                    if cost.is_finite() && cost >= 0.0 && cost < directed_cost[slot] {
                        directed_cost[slot] = cost;
                    }
                }
            }
        }

        // Build a robust undirected matrix for spanning-tree and degree
        // estimates. Missing directions fall back to the available reverse
        // direction; completely missing edges contribute zero rather than
        // making the heuristic unusable on an unexpectedly compiled task.
        let mut symmetric_cost = vec![0.0; node_count * node_count];
        for i in 0..node_count {
            for j in 0..node_count {
                let forward = directed_cost[i * node_count + j];
                let reverse = directed_cost[j * node_count + i];
                symmetric_cost[i * node_count + j] = if forward.is_finite() && reverse.is_finite() {
                    forward.min(reverse)
                } else if forward.is_finite() {
                    forward
                } else if reverse.is_finite() {
                    reverse
                } else {
                    0.0
                };
            }
        }

        Ok(Self {
            customers,
            depot_loc_fact: depot_name
                .as_ref()
                .and_then(|name| loc_by_name.get(name).cloned()),
            directed_cost,
            symmetric_cost,
            node_count,
        })
    }

    #[inline]
    fn directed(&self, from: usize, to: usize) -> f64 {
        let value = self.directed_cost[from * self.node_count + to];
        if value.is_finite() {
            value.max(0.0)
        } else {
            self.symmetric_cost[from * self.node_count + to].max(0.0)
        }
    }

    #[inline]
    fn symmetric(&self, a: usize, b: usize) -> f64 {
        self.symmetric_cost[a * self.node_count + b].max(0.0)
    }

    fn nearest_neighbor_route(
        &self,
        current: usize,
        unvisited: &[usize],
        depot_bias: f64,
    ) -> f64 {
        if unvisited.is_empty() {
            return self.directed(current, 0);
        }

        let mut remaining = vec![true; unvisited.len()];
        let mut left = unvisited.len();
        let mut position = current;
        let mut total = 0.0;

        while left > 0 {
            let mut best_index = None;
            let mut best_score = f64::INFINITY;

            for (index, &customer) in unvisited.iter().enumerate() {
                if !remaining[index] {
                    continue;
                }

                let edge = self.directed(position, customer);
                let score = edge + depot_bias * self.directed(customer, 0);
                if score < best_score {
                    best_score = score;
                    best_index = Some(index);
                }
            }

            let Some(index) = best_index else {
                break;
            };
            let next = unvisited[index];
            total += self.directed(position, next);
            position = next;
            remaining[index] = false;
            left -= 1;
        }

        total + self.directed(position, 0)
    }

    fn mst_estimate(&self, nodes: &[usize]) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let n = nodes.len();
        let mut in_tree = vec![false; n];
        let mut best = vec![f64::INFINITY; n];
        best[0] = 0.0;
        let mut total = 0.0;

        for _ in 0..n {
            let mut chosen = None;
            let mut chosen_cost = f64::INFINITY;
            for i in 0..n {
                if !in_tree[i] && best[i] < chosen_cost {
                    chosen = Some(i);
                    chosen_cost = best[i];
                }
            }

            let Some(u) = chosen else {
                break;
            };
            in_tree[u] = true;
            total += chosen_cost.max(0.0);

            for v in 0..n {
                if !in_tree[v] {
                    let edge = self.symmetric(nodes[u], nodes[v]);
                    if edge < best[v] {
                        best[v] = edge;
                    }
                }
            }
        }

        total
    }

    fn degree_estimate(&self, nodes: &[usize], current: usize) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let same_endpoint = current == 0;
        let mut sum = 0.0;

        for (i, &node) in nodes.iter().enumerate() {
            let mut first = f64::INFINITY;
            let mut second = f64::INFINITY;

            for (j, &other) in nodes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let edge = self.symmetric(node, other);
                if edge < first {
                    second = first;
                    first = edge;
                } else if edge < second {
                    second = edge;
                }
            }

            let endpoint = !same_endpoint && (node == current || node == 0);
            if endpoint {
                if first.is_finite() {
                    sum += first;
                }
            } else {
                if first.is_finite() {
                    sum += first;
                }
                if second.is_finite() {
                    sum += second;
                } else if first.is_finite() {
                    sum += first;
                }
            }
        }

        0.5 * sum
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

        let mut current = 0usize;
        if let Some(fact) = &self.depot_loc_fact {
            if fact.is_hold(registry.view(state)) {
                current = 0;
            }
        }

        let mut unvisited = Vec::with_capacity(self.customers.len());
        for (customer_id, customer) in self.customers.iter().enumerate() {
            let node = customer_id + 1;
            if !customer.visited_fact.is_hold(registry.view(state)) {
                unvisited.push(node);
            }
            if let Some(loc_fact) = &customer.loc_fact {
                if loc_fact.is_hold(registry.view(state)) {
                    current = node;
                }
            }
        }

        // Once every customer is served, the sole remaining routing action is
        // the final return to the depot.
        if unvisited.is_empty() {
            return Ok(self.directed(current, 0));
        }

        // Two inexpensive constructive completions are evaluated. The second
        // mildly favors leaving depot-near customers for the end of the route.
        let plain_nn = self.nearest_neighbor_route(current, &unvisited, 0.0);
        let return_aware_nn = self.nearest_neighbor_route(current, &unvisited, -0.15);
        let route_estimate = plain_nn.min(return_aware_nn);

        let mut relaxation_nodes = Vec::with_capacity(unvisited.len() + 2);
        relaxation_nodes.push(0);
        if current != 0 {
            relaxation_nodes.push(current);
        }
        relaxation_nodes.extend_from_slice(&unvisited);

        let mst = self.mst_estimate(&relaxation_nodes);
        let degree = self.degree_estimate(&relaxation_nodes, current);
        let structural_estimate = mst.max(degree);

        // The constructive estimate supplies strong ordering information while
        // the structural estimates reduce sensitivity to nearest-neighbor
        // choices. A small inflation compensates for the omitted capacity-driven
        // depot returns; admissibility is not required for this heuristic.
        let mut h = 1.15 * (0.80 * route_estimate + 0.20 * structural_estimate);
        if !h.is_finite() || h < 0.0 {
            h = 0.0;
        }

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
