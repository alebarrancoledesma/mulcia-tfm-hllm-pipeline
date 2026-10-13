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
use std::collections::HashMap;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

#[derive(Clone, Copy)]
struct PropFactRef {
    var_id: usize,
    positive_value: usize,
}

struct CustomerInfo {
    visited: PropFactRef,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    remaining: Vec<bool>,
    path: Vec<usize>,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    locations: Vec<PropFactRef>,
    depot: usize,
    direct_costs: Vec<f64>,
    via_depot_costs: Vec<f64>,
    node_count: usize,
    scratch: RefCell<ScratchBuffers>,
}

fn parse_positive_fact<'a>(fact_name: &'a str, predicate: &str) -> Option<&'a str> {
    let prefix = format!("Atom {predicate}(");
    fact_name
        .strip_prefix(&prefix)
        .and_then(|rest| rest.strip_suffix(')'))
        .map(str::trim)
        .filter(|argument| !argument.is_empty())
}

fn finite_cost(cost: f64) -> Option<f64> {
    if cost.is_finite() && cost >= 0.0 {
        Some(cost)
    } else {
        None
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Numeric IDs are deliberately not needed by this routing relaxation.
        // Still perform the documented one-time numeric-variable traversal so
        // that future extensions do not need to inspect metadata per state.
        for (_id, var) in task.numeric_variables().iter().enumerate() {
            let _ = var.name();
        }

        let mut visited_facts: HashMap<String, PropFactRef> = HashMap::new();
        let mut location_facts: HashMap<String, PropFactRef> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let var = &task.variables()[var_id];
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = parse_positive_fact(&fact_name, "visited") {
                    visited_facts.insert(
                        customer.to_string(),
                        PropFactRef {
                            var_id,
                            positive_value: value,
                        },
                    );
                }

                if let Some(place) = parse_positive_fact(&fact_name, "loc") {
                    location_facts.insert(
                        place.to_string(),
                        PropFactRef {
                            var_id,
                            positive_value: value,
                        },
                    );
                }
            }
        }

        if visited_facts.is_empty() {
            return Err(
                "llm_001 could not discover any positive visited(customer) facts"
                    .to_string()
                    .into(),
            );
        }

        let mut customer_names: Vec<String> = visited_facts.keys().cloned().collect();
        customer_names.sort();

        let mut depot_candidates: Vec<String> = location_facts
            .keys()
            .filter(|name| !visited_facts.contains_key(*name))
            .cloned()
            .collect();
        depot_candidates.sort();

        if depot_candidates.len() != 1 {
            return Err(format!(
                "llm_001 expected exactly one non-customer location for the depot, found {}",
                depot_candidates.len()
            )
            .into());
        }

        let depot_name = depot_candidates[0].clone();
        let customer_count = customer_names.len();
        let depot = customer_count;
        let node_count = customer_count + 1;

        let mut node_ids: HashMap<String, usize> = HashMap::new();
        for (id, name) in customer_names.iter().enumerate() {
            node_ids.insert(name.clone(), id);
        }
        node_ids.insert(depot_name.clone(), depot);

        let mut customers = Vec::with_capacity(customer_count);
        let mut locations = Vec::with_capacity(node_count);

        for name in &customer_names {
            let visited = match visited_facts.get(name) {
                Some(fact) => *fact,
                None => {
                    return Err(format!(
                        "llm_001 lost the visited fact mapping for customer {name}"
                    )
                    .into())
                }
            };
            let location = match location_facts.get(name) {
                Some(fact) => *fact,
                None => {
                    return Err(format!(
                        "llm_001 could not discover loc({name})"
                    )
                    .into())
                }
            };
            customers.push(CustomerInfo { visited });
            locations.push(location);
        }

        let depot_location = match location_facts.get(&depot_name) {
            Some(fact) => *fact,
            None => {
                return Err(
                    "llm_001 could not discover the depot location fact"
                        .to_string()
                        .into(),
                )
            }
        };
        locations.push(depot_location);

        let matrix_size = node_count * node_count;
        let mut direct_costs = vec![f64::INFINITY; matrix_size];
        let mut via_depot_costs = vec![f64::INFINITY; matrix_size];
        for node in 0..node_count {
            direct_costs[node * node_count + node] = 0.0;
            via_depot_costs[node * node_count + node] = 0.0;
        }

        for op in task.get_operators() {
            let parts: Vec<&str> = op.name().split_whitespace().collect();
            let cost = match finite_cost(metric_operator_cost_from_initial_values(task, op)) {
                Some(value) => value,
                None => continue,
            };

            match parts.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_ids.get(*from), node_ids.get(*to))
                    {
                        let index = from_id * node_count + to_id;
                        if cost < direct_costs[index] {
                            direct_costs[index] = cost;
                        }
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_ids.get(*from), node_ids.get(*to))
                    {
                        let index = from_id * node_count + to_id;
                        if cost < via_depot_costs[index] {
                            via_depot_costs[index] = cost;
                        }
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = node_ids.get(*from) {
                        let index = from_id * node_count + depot;
                        if cost < direct_costs[index] {
                            direct_costs[index] = cost;
                        }
                    }
                }
                _ => {}
            }
        }

        for from in 0..node_count {
            for to in 0..customer_count {
                if from == to {
                    continue;
                }
                let cost = direct_costs[from * node_count + to];
                if !cost.is_finite() {
                    return Err(format!(
                        "llm_001 could not reconstruct the visit cost from node {from} to customer {to}"
                    )
                    .into());
                }
            }
        }

        for customer in 0..customer_count {
            let cost = direct_costs[customer * node_count + depot];
            if !cost.is_finite() {
                return Err(format!(
                    "llm_001 could not reconstruct the return-to-depot cost for customer {customer}"
                )
                .into());
            }
        }

        Ok(Self {
            customers,
            locations,
            depot,
            direct_costs,
            via_depot_costs,
            node_count,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                remaining: vec![false; customer_count],
                path: Vec::with_capacity(node_count + 1),
            }),
        })
    }

    fn edge_cost(&self, from: usize, to: usize) -> f64 {
        let index = from * self.node_count + to;
        let direct = self.direct_costs[index];

        if from == self.depot || to == self.depot {
            direct
        } else {
            direct.min(self.via_depot_costs[index])
        }
    }

    fn path_cost(&self, path: &[usize]) -> f64 {
        let mut total = 0.0;
        for edge in path.windows(2) {
            let cost = self.edge_cost(edge[0], edge[1]);
            if !cost.is_finite() || cost < 0.0 {
                return f64::INFINITY;
            }
            total += cost;
        }
        total
    }

    fn improve_by_relocation(&self, path: &mut Vec<usize>) {
        if path.len() < 4 {
            return;
        }

        // A few best-improvement relocation passes substantially repair the
        // myopia of nearest-neighbour construction while retaining O(n^2)
        // evaluation time and fixed start/end nodes.
        for _ in 0..3 {
            let mut best_delta = -1.0e-9;
            let mut best_move: Option<(usize, usize)> = None;

            for i in 1..path.len() - 1 {
                let a = path[i - 1];
                let x = path[i];
                let b = path[i + 1];
                let removal = self.edge_cost(a, b)
                    - self.edge_cost(a, x)
                    - self.edge_cost(x, b);

                if !removal.is_finite() {
                    continue;
                }

                for j in 0..path.len() - 1 {
                    // The insertion edge must remain after x is removed.
                    if j == i || j + 1 == i {
                        continue;
                    }

                    let c = path[j];
                    let d = path[j + 1];
                    let delta = removal
                        + self.edge_cost(c, x)
                        + self.edge_cost(x, d)
                        - self.edge_cost(c, d);

                    if delta.is_finite() && delta < best_delta {
                        best_delta = delta;
                        best_move = Some((i, j));
                    }
                }
            }

            let (remove_index, original_left_index) = match best_move {
                Some(movement) => movement,
                None => break,
            };

            let node = path.remove(remove_index);
            let insertion_index = if original_left_index < remove_index {
                original_left_index + 1
            } else {
                original_left_index
            };
            path.insert(insertion_index, node);
        }
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
        let mut scratch = self.scratch.borrow_mut();

        state.fill_state(registry, &mut scratch.propositional);

        let mut current_location: Option<usize> = None;
        for (node, fact_ref) in self.locations.iter().enumerate() {
            if scratch.propositional[fact_ref.var_id] == fact_ref.positive_value {
                current_location = Some(node);
                break;
            }
        }

        let current = current_location.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "LLM heuristic could not identify the current CVRP location"
                    .to_string(),
            )
        })?;

        let mut unvisited_count = 0usize;
        for customer in 0..self.customers.len() {
            let fact_ref = self.customers[customer].visited;
            let visited =
                scratch.propositional[fact_ref.var_id] == fact_ref.positive_value;
            scratch.remaining[customer] = !visited;
            if !visited {
                unvisited_count += 1;
            }
        }

        if unvisited_count == 0 {
            let cost = self.edge_cost(current, self.depot);
            return Ok(if cost.is_finite() && cost >= 0.0 {
                cost
            } else {
                0.0
            });
        }

        scratch.path.clear();
        scratch.path.push(current);
        let mut cursor = current;

        // Construct a complete relaxed route by repeatedly choosing the
        // cheapest remaining transition. Customer-to-customer transitions may
        // use either a direct visit or the grounded visit-via-depot action.
        for _ in 0..unvisited_count {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for customer in 0..self.customers.len() {
                if !scratch.remaining[customer] {
                    continue;
                }
                let cost = self.edge_cost(cursor, customer);
                if cost < best_cost {
                    best_cost = cost;
                    best_customer = Some(customer);
                }
            }

            let next = best_customer.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "LLM heuristic could not extend its relaxed CVRP route"
                        .to_string(),
                )
            })?;

            scratch.remaining[next] = false;
            scratch.path.push(next);
            cursor = next;
        }

        scratch.path.push(self.depot);
        self.improve_by_relocation(&mut scratch.path);

        let estimate = self.path_cost(&scratch.path);
        let h = if estimate.is_finite() && estimate >= 0.0 {
            estimate
        } else {
            0.0
        };

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
