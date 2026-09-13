use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::task::ExplicitFact;

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

#[derive(Clone, Copy)]
struct PropFactRef {
    var_id: usize,
    positive_value: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    selected: Vec<bool>,
    mst_key: Vec<f64>,
    in_mst: Vec<bool>,
}

struct LlmHeuristic {
    // Node zero is the depot; nodes 1.. are customers.
    customer_visited: Vec<PropFactRef>,
    node_locations: Vec<PropFactRef>,
    costs: Vec<Vec<f64>>,
    scratch: RefCell<ScratchBuffers>,
}

fn positive_atom_argument(name: &str, predicate: &str) -> Option<String> {
    let prefix = format!("Atom {}(", predicate);
    if !name.starts_with(&prefix) || !name.ends_with(')') {
        return None;
    }

    let argument = &name[prefix.len()..name.len() - 1];
    if argument.is_empty() || argument.chars().any(char::is_whitespace) {
        return None;
    }

    Some(argument.to_string())
}

fn finite_nonnegative(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut visited_facts: HashMap<String, PropFactRef> = HashMap::new();
        let mut location_facts: HashMap<String, PropFactRef> = HashMap::new();

        // Discover semantic propositions once. Propositional variable names are
        // intentionally not used because they are translator-generated.
        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(customer) = positive_atom_argument(&fact_name, "visited") {
                    visited_facts.insert(
                        customer,
                        PropFactRef {
                            var_id,
                            positive_value: value,
                        },
                    );
                }

                if let Some(place) = positive_atom_argument(&fact_name, "loc") {
                    location_facts.insert(
                        place,
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

        let customer_names: BTreeSet<String> = visited_facts.keys().cloned().collect();
        let depot_candidates: Vec<String> = location_facts
            .keys()
            .filter(|name| !customer_names.contains(*name))
            .cloned()
            .collect();

        if depot_candidates.len() != 1 {
            return Err(format!(
                "llm_001 expected exactly one non-customer location for the depot, found {}",
                depot_candidates.len()
            )
            .into());
        }

        let depot_name = depot_candidates[0].clone();
        let ordered_customers: Vec<String> = customer_names.into_iter().collect();

        let depot_location = match location_facts.get(&depot_name).copied() {
            Some(reference) => reference,
            None => {
                return Err(
                    "llm_001 discovered a depot name but no positive depot location fact"
                        .to_string()
                        .into(),
                )
            }
        };

        let mut customer_visited = Vec::with_capacity(ordered_customers.len());
        let mut node_locations = Vec::with_capacity(ordered_customers.len() + 1);
        node_locations.push(depot_location);

        for customer in &ordered_customers {
            let visited = match visited_facts.get(customer).copied() {
                Some(reference) => reference,
                None => {
                    return Err(format!(
                        "llm_001 lost the visited fact for customer {}",
                        customer
                    )
                    .into())
                }
            };
            let location = match location_facts.get(customer).copied() {
                Some(reference) => reference,
                None => {
                    return Err(format!(
                        "llm_001 could not find loc({})",
                        customer
                    )
                    .into())
                }
            };

            customer_visited.push(visited);
            node_locations.push(location);
        }

        let mut node_ids: HashMap<String, usize> = HashMap::new();
        node_ids.insert(depot_name, 0);
        for (index, customer) in ordered_customers.iter().enumerate() {
            node_ids.insert(customer.clone(), index + 1);
        }

        // A return-to-depot operator has exactly the routing-cost numeric
        // update in this domain. Its affected variable therefore identifies
        // the accumulated metric-cost variable without relying on numeric IDs
        // or translator-generated numeric expression names.
        let mut return_cost_targets: HashSet<usize> = HashSet::new();
        for op in task.get_operators() {
            let mut words = op.name().split_whitespace();
            if words.next() == Some("return-to-depot") {
                for assignment in op.assignment_effects() {
                    return_cost_targets.insert(assignment.affected_var_id());
                }
            }
        }

        if return_cost_targets.len() != 1 {
            return Err(format!(
                "llm_001 could not uniquely identify the routing cost variable from return-to-depot operators (found {} candidates)",
                return_cost_targets.len()
            )
            .into());
        }

        let cost_var_id = match return_cost_targets.iter().next().copied() {
            Some(id) => id,
            None => {
                return Err(
                    "llm_001 found no routing cost variable".to_string().into(),
                )
            }
        };

        let initial_numeric = task.get_initial_numeric_state_values();
        let node_count = ordered_customers.len() + 1;
        let mut costs = vec![vec![f64::INFINITY; node_count]; node_count];
        for i in 0..node_count {
            costs[i][i] = 0.0;
        }

        // Recover directed travel costs from grounded operator structure. For
        // visit A B this is A->B, while return-to-depot A supplies A->depot.
        for op in task.get_operators() {
            let words: Vec<&str> = op.name().split_whitespace().collect();
            let endpoints = match words.as_slice() {
                ["visit", from, to] => Some((*from, *to)),
                ["return-to-depot", from] => Some((*from, depot_candidates[0].as_str())),
                _ => None,
            };

            let (from_name, to_name) = match endpoints {
                Some(pair) => pair,
                None => continue,
            };

            let from = match node_ids.get(from_name).copied() {
                Some(index) => index,
                None => continue,
            };
            let to = match node_ids.get(to_name).copied() {
                Some(index) => index,
                None => continue,
            };

            let mut recovered_cost: Option<f64> = None;
            for assignment in op.assignment_effects() {
                if assignment.affected_var_id() != cost_var_id {
                    continue;
                }

                if let Some(value) = initial_numeric.get(assignment.var_id()).copied() {
                    if let Some(value) = finite_nonnegative(value) {
                        recovered_cost = Some(value);
                    }
                }
            }

            if let Some(value) = recovered_cost {
                costs[from][to] = value;
            }
        }

        for from in 0..node_count {
            for to in 0..node_count {
                if from != to && !costs[from][to].is_finite() {
                    return Err(format!(
                        "llm_001 could not recover directed routing cost {} -> {}",
                        from, to
                    )
                    .into());
                }
            }
        }

        Ok(Self {
            customer_visited,
            node_locations,
            costs,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                selected: vec![false; node_count],
                mst_key: vec![f64::INFINITY; node_count],
                in_mst: vec![false; node_count],
            }),
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
        let mut scratch = self.scratch.borrow_mut();
        state.fill_state(registry, &mut scratch.propositional);

        let current = self
            .node_locations
            .iter()
            .position(|reference| {
                scratch.propositional[reference.var_id] == reference.positive_value
            })
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "LLM heuristic could not determine the current CVRP location"
                        .to_string(),
                )
            })?;

        let mut remaining_count = 0usize;
        for (customer_index, reference) in self.customer_visited.iter().enumerate() {
            let node = customer_index + 1;
            let visited =
                scratch.propositional[reference.var_id] == reference.positive_value;
            scratch.selected[node] = visited;
            if !visited {
                remaining_count += 1;
            }
        }
        scratch.selected[0] = true;

        if remaining_count == 0 {
            let value = self.costs[current][0];
            return Ok(if value.is_finite() && value >= 0.0 {
                value
            } else {
                0.0
            });
        }

        // Fast directed nearest-neighbor completion from the current location.
        // This gives a concrete one-route estimate and strongly distinguishes
        // states according to the spatial coherence of their remaining tour.
        let mut nearest_neighbor = 0.0;
        let mut cursor = current;
        for _ in 0..remaining_count {
            let mut best_node: Option<usize> = None;
            let mut best_cost = f64::INFINITY;

            for node in 1..self.costs.len() {
                if !scratch.selected[node] && self.costs[cursor][node] < best_cost {
                    best_cost = self.costs[cursor][node];
                    best_node = Some(node);
                }
            }

            let node = match best_node {
                Some(node) => node,
                None => break,
            };

            nearest_neighbor += best_cost;
            scratch.selected[node] = true;
            cursor = node;
        }
        nearest_neighbor += self.costs[cursor][0];

        // Recreate the unvisited mask and compute a Prim MST over customers.
        // For the undirected relaxation, the cheaper of the two directed arcs
        // is used. Endpoint connections account for leaving the current node
        // and eventually returning to the depot.
        for (customer_index, reference) in self.customer_visited.iter().enumerate() {
            let node = customer_index + 1;
            let visited =
                scratch.propositional[reference.var_id] == reference.positive_value;
            scratch.in_mst[node] = visited;
            scratch.mst_key[node] = f64::INFINITY;
        }
        scratch.in_mst[0] = true;
        scratch.mst_key[0] = f64::INFINITY;

        let first_unvisited = (1..self.costs.len()).find(|&node| !scratch.in_mst[node]);
        let mut mst_cost = 0.0;

        if let Some(first) = first_unvisited {
            scratch.mst_key[first] = 0.0;

            for _ in 0..remaining_count {
                let mut chosen: Option<usize> = None;
                let mut chosen_key = f64::INFINITY;

                for node in 1..self.costs.len() {
                    if !scratch.in_mst[node] && scratch.mst_key[node] < chosen_key {
                        chosen = Some(node);
                        chosen_key = scratch.mst_key[node];
                    }
                }

                let node = match chosen {
                    Some(node) => node,
                    None => break,
                };

                scratch.in_mst[node] = true;
                mst_cost += chosen_key;

                for other in 1..self.costs.len() {
                    if scratch.in_mst[other] {
                        continue;
                    }
                    let relaxed_edge =
                        self.costs[node][other].min(self.costs[other][node]);
                    if relaxed_edge < scratch.mst_key[other] {
                        scratch.mst_key[other] = relaxed_edge;
                    }
                }
            }
        }

        let mut cheapest_departure = f64::INFINITY;
        let mut cheapest_return = f64::INFINITY;
        for (customer_index, reference) in self.customer_visited.iter().enumerate() {
            let node = customer_index + 1;
            let visited =
                scratch.propositional[reference.var_id] == reference.positive_value;
            if !visited {
                cheapest_departure = cheapest_departure.min(self.costs[current][node]);
                cheapest_return = cheapest_return.min(self.costs[node][0]);
            }
        }

        let spanning_estimate = mst_cost + cheapest_departure + cheapest_return;

        // The nearest-neighbor estimate captures route shape, while the MST
        // endpoint estimate is more stable under individual customer choices.
        let h = 0.70 * nearest_neighbor + 0.30 * spanning_estimate;

        if h.is_finite() && h >= 0.0 {
            Ok(h)
        } else {
            Ok(0.0)
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
                .into(),
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
        },
    ])
}
