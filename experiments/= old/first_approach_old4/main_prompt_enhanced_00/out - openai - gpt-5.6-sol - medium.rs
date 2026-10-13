use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Copy)]
struct FactSlot {
    variable: usize,
    value: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    used_nodes: Vec<bool>,
}

struct LlmHeuristic {
    // Node zero is the depot; nodes 1.. are customers.
    customer_visited: Vec<FactSlot>,
    node_locations: Vec<Option<FactSlot>>,
    costs: Vec<f64>,
    node_count: usize,
    fallback_cost: f64,
    scratch: RefCell<ScratchBuffers>,
}

fn parse_atom_object<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let prefix = if predicate == "visited" {
        "Atom visited("
    } else if predicate == "loc" {
        "Atom loc("
    } else {
        return None;
    };

    name.strip_prefix(prefix)?.strip_suffix(')')
}

fn finite_nonnegative_cost(cost: f64) -> Option<f64> {
    if cost.is_finite() && cost >= 0.0 {
        Some(cost.min(1.0e100))
    } else {
        None
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut visited_by_name: HashMap<String, FactSlot> = HashMap::new();
        let mut location_by_name: HashMap<String, FactSlot> = HashMap::new();
        let mut all_location_names: Vec<String> = Vec::new();

        for variable in 0..task.variables().len() {
            let domain_size = task.variables()[variable].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = parse_atom_object(fact_name, "visited") {
                    visited_by_name.insert(
                        object.to_string(),
                        FactSlot { variable, value },
                    );
                }

                if let Some(object) = parse_atom_object(fact_name, "loc") {
                    if !location_by_name.contains_key(object) {
                        all_location_names.push(object.to_string());
                    }
                    location_by_name.insert(
                        object.to_string(),
                        FactSlot { variable, value },
                    );
                }
            }
        }

        let customer_names_set: HashSet<String> =
            visited_by_name.keys().cloned().collect();

        // In this domain the unique place with a location fact but no visited
        // predicate is the depot. If compilation removed that distinction,
        // recover it from the source argument of a depot-to-customer visit.
        let mut depot_name = all_location_names
            .iter()
            .find(|name| !customer_names_set.contains(*name))
            .cloned();

        if depot_name.is_none() {
            for op in task.get_operators() {
                let parts: Vec<&str> = op.name().split_whitespace().collect();
                if parts.len() == 3
                    && parts[0] == "visit"
                    && !customer_names_set.contains(parts[1])
                    && customer_names_set.contains(parts[2])
                {
                    depot_name = Some(parts[1].to_string());
                    break;
                }
            }
        }

        let depot_name = match depot_name {
            Some(name) => name,
            None => {
                return Err(
                    "llm_001 could not identify the CVRP depot from location facts or visit operators"
                        .to_string()
                        .into(),
                );
            }
        };

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());

        let node_count = node_names.len();
        let mut node_id: HashMap<String, usize> = HashMap::new();
        for (id, name) in node_names.iter().enumerate() {
            node_id.insert(name.clone(), id);
        }

        let mut customer_visited = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            if let Some(slot) = visited_by_name.get(name) {
                customer_visited.push(*slot);
            } else {
                return Err(
                    format!("llm_001 found customer {name} without a visited fact").into(),
                );
            }
        }

        let mut node_locations = Vec::with_capacity(node_count);
        for name in &node_names {
            node_locations.push(location_by_name.get(name).copied());
        }

        let mut costs = vec![f64::NAN; node_count.saturating_mul(node_count)];
        for i in 0..node_count {
            costs[i * node_count + i] = 0.0;
        }

        let mut max_observed_cost: f64 = 1.0;

        for op in task.get_operators() {
            let parts: Vec<&str> = op.name().split_whitespace().collect();
            let endpoints = if parts.len() == 3 && parts[0] == "visit" {
                match (node_id.get(parts[1]), node_id.get(parts[2])) {
                    (Some(&from), Some(&to)) => Some((from, to)),
                    _ => None,
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                match (node_id.get(parts[1]), node_id.get(&node_names[0])) {
                    (Some(&from), Some(&to)) => Some((from, to)),
                    _ => None,
                }
            } else {
                None
            };

            if let Some((from, to)) = endpoints {
                let raw_cost = metric_operator_cost_from_initial_values(task, op);
                if let Some(cost) = finite_nonnegative_cost(raw_cost) {
                    let index = from * node_count + to;
                    if !costs[index].is_finite() || cost < costs[index] {
                        costs[index] = cost;
                    }
                    max_observed_cost = max_observed_cost.max(cost);
                }
            }
        }

        let scale = (node_count as f64 + 1.0).max(1.0);
        let fallback_cost = (max_observed_cost * scale).min(1.0e100).max(1.0);

        // Complete missing arcs conservatively with a large finite value. The
        // standard CVRP grounding supplies all such visit and return arcs.
        for cost in &mut costs {
            if !cost.is_finite() || *cost < 0.0 {
                *cost = fallback_cost;
            }
        }

        Ok(Self {
            customer_visited,
            node_locations,
            costs,
            node_count,
            fallback_cost,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                used_nodes: vec![false; node_count],
            }),
        })
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        if from < self.node_count && to < self.node_count {
            self.costs[from * self.node_count + to]
        } else {
            self.fallback_cost
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
        scratch.used_nodes.fill(false);
        if !scratch.used_nodes.is_empty() {
            scratch.used_nodes[0] = true;
        }

        let mut remaining = 0usize;
        for customer_index in 0..self.customer_visited.len() {
            let slot = self.customer_visited[customer_index];
            let visited = scratch
                .propositional
                .get(slot.variable)
                .map(|&value| value == slot.value)
                .unwrap_or(false);

            let node = customer_index + 1;
            if visited {
                if node < scratch.used_nodes.len() {
                    scratch.used_nodes[node] = true;
                }
            } else {
                remaining += 1;
            }
        }

        let mut current_node = 0usize;
        for (node, location) in self.node_locations.iter().enumerate() {
            if let Some(slot) = location {
                let holds = scratch
                    .propositional
                    .get(slot.variable)
                    .map(|&value| value == slot.value)
                    .unwrap_or(false);
                if holds {
                    current_node = node;
                    break;
                }
            }
        }

        // Build a directed nearest-neighbour completion from the current
        // vehicle location through all unvisited customers and back to depot.
        let mut h = 0.0;
        while remaining > 0 {
            let mut best_node: Option<usize> = None;
            let mut best_cost = self.fallback_cost;

            for node in 1..self.node_count {
                if !scratch.used_nodes[node] {
                    let edge_cost = self.cost(current_node, node);
                    if best_node.is_none() || edge_cost < best_cost {
                        best_node = Some(node);
                        best_cost = edge_cost;
                    }
                }
            }

            let next = match best_node {
                Some(node) => node,
                None => break,
            };

            h += best_cost;
            if !h.is_finite() || h >= 1.0e100 {
                return Ok(1.0e100);
            }

            scratch.used_nodes[next] = true;
            current_node = next;
            remaining -= 1;
        }

        h += self.cost(current_node, 0);

        if h.is_finite() && h >= 0.0 {
            Ok(h.min(1.0e100))
        } else {
            Ok(self.fallback_cost)
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
