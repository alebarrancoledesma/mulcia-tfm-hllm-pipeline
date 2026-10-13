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

#[derive(Clone)]
struct NamedArc {
    from: String,
    to: Option<String>,
    cost: f64,
}

struct CustomerInfo {
    visited_var: usize,
    visited_value: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    unvisited: Vec<usize>,
    active: Vec<usize>,
    key: Vec<f64>,
    used: Vec<bool>,
}

struct LlmHeuristic {
    node_count: usize,
    depot: usize,
    customers: Vec<CustomerInfo>,
    location_facts: Vec<(usize, usize)>,
    direct_costs: Vec<f64>,
    scratch: RefCell<ScratchBuffers>,
}

impl LlmHeuristic {
    fn extract_argument(fact_name: &str, predicate: &str) -> Option<String> {
        let marker = format!("{}(", predicate);
        let start = fact_name.find(&marker)? + marker.len();
        let rest = &fact_name[start..];
        let end = rest.find(')')?;
        let argument = rest[..end].trim();
        if argument.is_empty() || argument.contains(',') {
            None
        } else {
            Some(argument.to_string())
        }
    }

    fn valid_cost(cost: f64) -> Option<f64> {
        if cost.is_finite() && cost >= 0.0 {
            Some(cost)
        } else {
            None
        }
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.direct_costs[from * self.node_count + to]
    }

    fn transition_cost(&self, from: usize, to: usize) -> f64 {
        let direct = self.cost(from, to);
        if from == self.depot || to == self.depot {
            return direct;
        }

        let via_depot = self.cost(from, self.depot) + self.cost(self.depot, to);
        direct.min(via_depot)
    }

    fn undirected_connection_cost(&self, a: usize, b: usize) -> f64 {
        self.transition_cost(a, b)
            .min(self.transition_cost(b, a))
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names = HashSet::new();
        let mut visit_arcs = Vec::new();
        let mut return_arcs = Vec::new();

        for op in task.get_operators() {
            let parts: Vec<&str> = op.name().split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let cost = match Self::valid_cost(
                metric_operator_cost_from_initial_values(task, op),
            ) {
                Some(value) => value,
                None => continue,
            };

            match parts[0] {
                "visit" if parts.len() == 3 => {
                    let from = parts[1].to_string();
                    let to = parts[2].to_string();
                    customer_names.insert(to.clone());
                    visit_arcs.push(NamedArc {
                        from,
                        to: Some(to),
                        cost,
                    });
                }
                "visit-via-depot" if parts.len() == 3 => {
                    customer_names.insert(parts[1].to_string());
                    customer_names.insert(parts[2].to_string());
                }
                "return-to-depot" if parts.len() == 2 => {
                    let from = parts[1].to_string();
                    customer_names.insert(from.clone());
                    return_arcs.push(NamedArc {
                        from,
                        to: None,
                        cost,
                    });
                }
                _ => {}
            }
        }

        if customer_names.is_empty() {
            return Err(
                "llm_001 could not discover any CVRP customers from grounded operators"
                    .to_string()
                    .into(),
            );
        }

        let mut depot_candidates = HashSet::new();
        for arc in &visit_arcs {
            if !customer_names.contains(&arc.from) {
                depot_candidates.insert(arc.from.clone());
            }
        }

        if depot_candidates.len() != 1 {
            return Err(
                format!(
                    "llm_001 expected one depot source in visit operators, found {}",
                    depot_candidates.len()
                )
                .into(),
            );
        }

        let depot_name = match depot_candidates.into_iter().next() {
            Some(name) => name,
            None => {
                return Err(
                    "llm_001 could not identify the depot"
                        .to_string()
                        .into(),
                );
            }
        };

        let mut ordered_customers: Vec<String> = customer_names.into_iter().collect();
        ordered_customers.sort();

        let mut node_names = Vec::with_capacity(ordered_customers.len() + 1);
        node_names.push(depot_name.clone());
        node_names.extend(ordered_customers.iter().cloned());

        let node_count = node_names.len();
        let depot = 0;
        let mut node_ids = HashMap::new();
        for (id, name) in node_names.iter().enumerate() {
            node_ids.insert(name.clone(), id);
        }

        let mut direct_costs = vec![f64::NAN; node_count * node_count];
        for id in 0..node_count {
            direct_costs[id * node_count + id] = 0.0;
        }

        let mut observed_cost_sum = 0.0;
        let mut observed_cost_count = 0usize;

        for arc in &visit_arcs {
            let to_name = match &arc.to {
                Some(name) => name,
                None => continue,
            };
            let from = match node_ids.get(&arc.from) {
                Some(id) => *id,
                None => continue,
            };
            let to = match node_ids.get(to_name) {
                Some(id) => *id,
                None => continue,
            };
            let index = from * node_count + to;
            if !direct_costs[index].is_finite() || arc.cost < direct_costs[index] {
                direct_costs[index] = arc.cost;
            }
            observed_cost_sum += arc.cost;
            observed_cost_count += 1;
        }

        for arc in &return_arcs {
            let from = match node_ids.get(&arc.from) {
                Some(id) => *id,
                None => continue,
            };
            let index = from * node_count + depot;
            if !direct_costs[index].is_finite() || arc.cost < direct_costs[index] {
                direct_costs[index] = arc.cost;
            }
            observed_cost_sum += arc.cost;
            observed_cost_count += 1;
        }

        let fallback_cost = if observed_cost_count > 0 {
            let mean = observed_cost_sum / observed_cost_count as f64;
            if mean.is_finite() && mean > 0.0 {
                mean * 4.0
            } else {
                1.0
            }
        } else {
            1.0
        };

        for value in &mut direct_costs {
            if !value.is_finite() || *value < 0.0 {
                *value = fallback_cost;
            }
        }

        let mut visited_facts: HashMap<String, (usize, usize)> = HashMap::new();
        let mut location_by_name: HashMap<String, (usize, usize)> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(name) = Self::extract_argument(fact_name, "visited") {
                    if node_ids.contains_key(&name) {
                        visited_facts.entry(name).or_insert((var_id, value));
                    }
                }

                if let Some(name) = Self::extract_argument(fact_name, "loc") {
                    if node_ids.contains_key(&name) {
                        location_by_name.entry(name).or_insert((var_id, value));
                    }
                }
            }
        }

        let mut customers = Vec::with_capacity(ordered_customers.len());
        for name in &ordered_customers {
            let (visited_var, visited_value) = match visited_facts.get(name) {
                Some(pair) => *pair,
                None => {
                    return Err(
                        format!(
                            "llm_001 could not find the visited fact for customer {}",
                            name
                        )
                        .into(),
                    );
                }
            };
            customers.push(CustomerInfo {
                visited_var,
                visited_value,
            });
        }

        let mut location_facts = Vec::with_capacity(node_count);
        for name in &node_names {
            match location_by_name.get(name) {
                Some(pair) => location_facts.push(*pair),
                None => {
                    return Err(
                        format!(
                            "llm_001 could not find the location fact for node {}",
                            name
                        )
                        .into(),
                    );
                }
            }
        }

        Ok(Self {
            node_count,
            depot,
            customers,
            location_facts,
            direct_costs,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                unvisited: Vec::with_capacity(node_count),
                active: Vec::with_capacity(node_count + 1),
                key: Vec::with_capacity(node_count + 1),
                used: vec![false; node_count + 1],
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

        let mut current = self.depot;
        for (node, &(var_id, value)) in self.location_facts.iter().enumerate() {
            if var_id < scratch.propositional.len()
                && scratch.propositional[var_id] == value
            {
                current = node;
                break;
            }
        }

        scratch.unvisited.clear();
        for (customer_index, customer) in self.customers.iter().enumerate() {
            let is_visited = customer.visited_var < scratch.propositional.len()
                && scratch.propositional[customer.visited_var]
                    == customer.visited_value;
            if !is_visited {
                scratch.unvisited.push(customer_index + 1);
            }
        }

        if scratch.unvisited.is_empty() {
            let h = self.cost(current, self.depot);
            return Ok(if h.is_finite() && h >= 0.0 { h } else { 0.0 });
        }

        // Directed nearest-neighbour completion. Each transition may either be
        // direct or pass through the depot, matching the two movement modes of
        // the CVRP domain while relaxing capacity and vehicle-count limits.
        scratch.used.fill(false);
        let mut nn_cost = 0.0;
        let mut cursor = current;

        for _ in 0..scratch.unvisited.len() {
            let mut best_node = None;
            let mut best_cost = f64::INFINITY;

            for &candidate in &scratch.unvisited {
                if !scratch.used[candidate] {
                    let edge = self.transition_cost(cursor, candidate);
                    if edge < best_cost {
                        best_cost = edge;
                        best_node = Some(candidate);
                    }
                }
            }

            match best_node {
                Some(node) => {
                    nn_cost += best_cost;
                    scratch.used[node] = true;
                    cursor = node;
                }
                None => break,
            }
        }
        nn_cost += self.cost(cursor, self.depot);

        // A Prim MST over the current position, depot, and unvisited customers
        // supplies a stable global connectivity estimate. Directed transition
        // costs are symmetrized by taking the cheaper direction only for this
        // connectivity component.
        scratch.active.clear();
        scratch.active.push(self.depot);
        if current != self.depot {
            scratch.active.push(current);
        }
        for &node in &scratch.unvisited {
            if !scratch.active.contains(&node) {
                scratch.active.push(node);
            }
        }

        let active_len = scratch.active.len();
        scratch.key.clear();
        scratch.key.resize(active_len, f64::INFINITY);
        scratch.used.clear();
        scratch.used.resize(active_len, false);
        if active_len > 0 {
            scratch.key[0] = 0.0;
        }

        let mut mst_cost = 0.0;
        for _ in 0..active_len {
            let mut selected = None;
            let mut selected_key = f64::INFINITY;
            for i in 0..active_len {
                if !scratch.used[i] && scratch.key[i] < selected_key {
                    selected = Some(i);
                    selected_key = scratch.key[i];
                }
            }

            let u = match selected {
                Some(index) => index,
                None => break,
            };
            scratch.used[u] = true;
            mst_cost += selected_key;

            let node_u = scratch.active[u];
            for v in 0..active_len {
                if !scratch.used[v] {
                    let node_v = scratch.active[v];
                    let edge = self.undirected_connection_cost(node_u, node_v);
                    if edge < scratch.key[v] {
                        scratch.key[v] = edge;
                    }
                }
            }
        }

        // The greedy completion is state-sensitive, while the scaled MST is
        // smoother and captures the cost of connecting all remaining regions.
        let raw_h = 0.65 * nn_cost + 0.595 * mst_cost;
        let h = if raw_h.is_finite() && raw_h >= 0.0 {
            raw_h.min(1.0e100)
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
