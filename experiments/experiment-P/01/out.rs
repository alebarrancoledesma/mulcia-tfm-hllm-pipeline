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

const INF: f64 = 1.0e100;
const DEAD_END_PENALTY: f64 = 1.0e15;

fn atom_argument<'a>(fact_name: &'a str, predicate: &str) -> Option<&'a str> {
    let mut text = fact_name.trim();
    if let Some(rest) = text.strip_prefix("Atom ") {
        text = rest.trim();
    }

    let prefix = format!("{}(", predicate);
    let inside = text.strip_prefix(&prefix)?.strip_suffix(')')?.trim();
    if inside.is_empty() || inside.contains(',') {
        None
    } else {
        Some(inside)
    }
}

fn normalized_pne_name(name: &str) -> &str {
    let text = name.trim();
    if let Some(rest) = text.strip_prefix("PNE ") {
        rest.trim()
    } else {
        text
    }
}

fn exact_zero_arity_pne(name: &str, wanted: &str) -> bool {
    normalized_pne_name(name) == wanted
}

fn demand_argument(name: &str) -> Option<&str> {
    let text = normalized_pne_name(name);
    let inside = text.strip_prefix("demand(")?.strip_suffix(')')?.trim();
    if inside.is_empty() || inside.contains(',') {
        None
    } else {
        Some(inside)
    }
}

struct LlmHeuristic {
    // Node zero is the depot; nodes 1.. are customers.
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    costs: Vec<f64>,
    node_count: usize,

    // Resource data is deliberately optional because the translator may fold
    // capacity, demand, and max_vehicles into derived expressions.
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    demands: Option<Vec<f64>>,
}

impl LlmHeuristic {
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.costs[from * self.node_count + to]
    }

    fn undirected_cost(&self, a: usize, b: usize) -> f64 {
        self.cost(a, b).min(self.cost(b, a))
    }

    fn nearest_neighbor_completion(
        &self,
        current: usize,
        remaining: &[bool],
    ) -> f64 {
        let mut active = remaining.to_vec();
        let mut left = active.iter().filter(|&&x| x).count();
        let mut position = current;
        let mut result: f64 = 0.0;

        while left > 0 {
            let mut best_node: Option<usize> = None;
            let mut best_cost: f64 = INF;

            for customer in 1..self.node_count {
                if active[customer] {
                    let edge = self.cost(position, customer);
                    if edge < best_cost {
                        best_cost = edge;
                        best_node = Some(customer);
                    }
                }
            }

            let Some(next) = best_node else {
                return DEAD_END_PENALTY;
            };

            result += best_cost;
            position = next;
            active[next] = false;
            left -= 1;
        }

        result + self.cost(position, 0)
    }

    fn mst_completion_bound(
        &self,
        current: usize,
        remaining: &[bool],
    ) -> f64 {
        let mut nodes: Vec<usize> = Vec::with_capacity(self.node_count + 1);
        nodes.push(0);

        if current != 0 {
            nodes.push(current);
        }

        for customer in 1..self.node_count {
            if remaining[customer] && customer != current {
                nodes.push(customer);
            }
        }

        if nodes.len() <= 1 {
            return 0.0;
        }

        let count = nodes.len();
        let mut in_tree = vec![false; count];
        let mut key = vec![INF; count];
        key[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_key: f64 = INF;

            for i in 0..count {
                if !in_tree[i] && key[i] < selected_key {
                    selected = Some(i);
                    selected_key = key[i];
                }
            }

            let Some(u) = selected else {
                return DEAD_END_PENALTY;
            };

            in_tree[u] = true;
            total += selected_key;

            for v in 0..count {
                if !in_tree[v] {
                    let edge = self.undirected_cost(nodes[u], nodes[v]);
                    if edge < key[v] {
                        key[v] = edge;
                    }
                }
            }
        }

        total
    }

    fn capacity_aware_completion(
        &self,
        current: usize,
        remaining: &[bool],
        initial_load: f64,
        current_vehicles: Option<f64>,
        demands: &[f64],
        capacity: f64,
    ) -> f64 {
        let mut active = remaining.to_vec();
        let mut left = active.iter().filter(|&&x| x).count();
        let mut position = current;
        let mut load = if current == 0 { 0.0 } else { initial_load.max(0.0) };
        let mut result: f64 = 0.0;
        let mut restarts: f64 = 0.0;
        let tolerance = 1.0e-9_f64;

        while left > 0 {
            let mut best_node: Option<usize> = None;
            let mut best_score: f64 = INF;

            for customer in 1..self.node_count {
                if !active[customer] {
                    continue;
                }

                let demand = demands[customer];
                if demand > capacity + tolerance {
                    return DEAD_END_PENALTY;
                }

                if load + demand <= capacity + tolerance {
                    // Nearest-neighbor is augmented by a small depot-distance
                    // term. It prefers nearby customers while mildly delaying
                    // customers that are cheap to place at a route boundary.
                    let score = self.cost(position, customer)
                        + 0.08 * self.cost(customer, 0);
                    if score < best_score {
                        best_score = score;
                        best_node = Some(customer);
                    }
                }
            }

            if let Some(next) = best_node {
                result += self.cost(position, next);
                load += demands[next];
                position = next;
                active[next] = false;
                left -= 1;
                continue;
            }

            // No remaining customer fits in the active vehicle. The domain's
            // visit-via-depot action pays exactly the return-to-depot plus the
            // next depot-to-customer edge, so explicitly close this route.
            if position == 0 {
                return DEAD_END_PENALTY;
            }

            result += self.cost(position, 0);
            position = 0;
            load = 0.0;
            restarts += 1.0;
        }

        result += self.cost(position, 0);

        if let (Some(used), Some(limit)) = (current_vehicles, self.max_vehicles) {
            if used + restarts > limit + tolerance {
                return DEAD_END_PENALTY + result;
            }
        }

        result
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut visited_facts: Vec<ExplicitFact> = Vec::new();
        let mut depot_name: Option<String> = None;

        // Goals identify customers and the unique depot without relying on
        // object names or on finite-domain value encodings.
        for i in 0..task.get_num_goals() {
            let fact: ExplicitFact = task.get_goal_fact(i);
            let fact_name = task.get_fact_name(&fact);

            if let Some(customer) = atom_argument(fact_name, "visited") {
                customer_names.push(customer.to_string());
                visited_facts.push(fact);
            } else if let Some(depot) = atom_argument(fact_name, "loc") {
                depot_name = Some(depot.to_string());
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP task has no visited-customer goals".to_string(),
            ));
        }

        let node_count = customer_names.len() + 1;
        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), 0);
        for (i, name) in customer_names.iter().enumerate() {
            node_by_name.insert(name.clone(), i + 1);
        }

        let mut location_options: Vec<Option<ExplicitFact>> = vec![None; node_count];
        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(place) = atom_argument(fact_name, "loc") {
                    if let Some(&node) = node_by_name.get(place) {
                        location_options[node] = Some(fact);
                    }
                }
            }
        }

        let mut location_facts: Vec<ExplicitFact> = Vec::with_capacity(node_count);
        for (node, fact) in location_options.into_iter().enumerate() {
            let fact = fact.ok_or_else(|| -> HeuristicBuildError {
                format!("missing location fact for CVRP node index {node}").into()
            })?;
            location_facts.push(fact);
        }

        let mut costs = vec![INF; node_count * node_count];
        for i in 0..node_count {
            costs[i * node_count + i] = 0.0;
        }

        // Grounded operator names provide a stable semantic mapping from
        // objects to arcs; metric_operator_cost_from_initial_values evaluates
        // the translated numeric cost without parsing travel-cost PNEs.
        for op in task.get_operators() {
            let words: Vec<&str> = op.name().split_whitespace().collect();
            if words.is_empty() {
                continue;
            }

            let metric_cost = metric_operator_cost_from_initial_values(task, op);
            if !metric_cost.is_finite() || metric_cost < 0.0 {
                continue;
            }

            match words[0] {
                "visit" if words.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(words[1]), node_by_name.get(words[2]))
                    {
                        let index = from * node_count + to;
                        costs[index] = costs[index].min(metric_cost);
                    }
                }
                "return-to-depot" if words.len() == 2 => {
                    if let Some(&from) = node_by_name.get(words[1]) {
                        let index = from * node_count;
                        costs[index] = costs[index].min(metric_cost);
                    }
                }
                _ => {}
            }
        }

        // Depot arcs must exist in this domain. Customer-to-customer arcs are
        // normally complete; if translation removes one, safely substitute
        // the corresponding via-depot cost rather than losing routing h.
        for customer in 1..node_count {
            if costs[customer].ge(&INF) || costs[customer * node_count].ge(&INF) {
                return Err(HeuristicBuildError::from(
                    "missing required depot/customer routing operator cost".to_string(),
                ));
            }
        }

        for from in 1..node_count {
            for to in 1..node_count {
                if from != to {
                    let index = from * node_count + to;
                    if costs[index] >= INF {
                        costs[index] = costs[from * node_count] + costs[to];
                    }
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut max_vehicles: Option<f64> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; node_count];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if exact_zero_arity_pne(name, "load()") {
                load_id = Some(id);
            } else if exact_zero_arity_pne(name, "vehicles()") {
                vehicles_id = Some(id);
            } else if exact_zero_arity_pne(name, "capacity()") {
                capacity = initial_numeric.get(id).copied();
            } else if exact_zero_arity_pne(name, "max_vehicles()") {
                max_vehicles = initial_numeric.get(id).copied();
            } else if let Some(customer) = demand_argument(name) {
                if let Some(&node) = node_by_name.get(customer) {
                    demand_ids[node] = Some(id);
                }
            }
        }

        let demands = if capacity.is_some() && demand_ids[1..].iter().all(Option::is_some) {
            let mut values = vec![0.0; node_count];
            let mut complete = true;
            for node in 1..node_count {
                let id = demand_ids[node].unwrap();
                if let Some(&value) = initial_numeric.get(id) {
                    values[node] = value.max(0.0);
                } else {
                    complete = false;
                    break;
                }
            }
            if complete { Some(values) } else { None }
        } else {
            None
        };

        Ok(Self {
            visited_facts,
            location_facts,
            costs,
            node_count,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let view = registry.view(state);

        let mut current: usize = 0;
        for node in 0..self.node_count {
            if self.location_facts[node].is_hold(view) {
                current = node;
                break;
            }
        }

        let mut remaining = vec![false; self.node_count];
        let mut remaining_count: usize = 0;
        for (customer_index, fact) in self.visited_facts.iter().enumerate() {
            let node = customer_index + 1;
            if !fact.is_hold(view) {
                remaining[node] = true;
                remaining_count += 1;
            }
        }

        if remaining_count == 0 {
            return Ok(self.cost(current, 0).max(0.0));
        }

        if let (Some(capacity), Some(demands), Some(load_id)) =
            (self.capacity, self.demands.as_ref(), self.load_id)
        {
            if capacity.is_finite() && capacity > 0.0 {
                let mut numeric_values = vec![0.0; task.numeric_variables().len()];
                registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

                let load = numeric_values.get(load_id).copied().unwrap_or(0.0);
                let vehicles = self.vehicles_id
                    .and_then(|id| numeric_values.get(id).copied());

                // A sound aggregate fleet check catches states whose residual
                // demand cannot fit into the current and remaining vehicles.
                if let (Some(used), Some(limit)) = (vehicles, self.max_vehicles) {
                    let mut demand_sum: f64 = 0.0;
                    for customer in 1..self.node_count {
                        if remaining[customer] {
                            demand_sum += demands[customer];
                        }
                    }

                    let residual = (capacity - load.max(0.0)).max(0.0);
                    let extra_routes = if demand_sum <= residual + 1.0e-9 {
                        0.0
                    } else {
                        ((demand_sum - residual) / capacity).ceil()
                    };

                    if used + extra_routes > limit + 1.0e-9 {
                        return Ok(DEAD_END_PENALTY);
                    }
                }

                let h = self.capacity_aware_completion(
                    current,
                    &remaining,
                    load,
                    vehicles,
                    demands,
                    capacity,
                );
                return Ok(h.max(0.0));
            }
        }

        // When folded resource PNEs are unavailable, preserve a strong pure
        // routing estimate. The greedy completion approximates the remaining
        // Hamiltonian path, while the MST component stabilizes the estimate
        // against unlucky nearest-neighbor choices.
        let greedy = self.nearest_neighbor_completion(current, &remaining);
        let mst = self.mst_completion_bound(current, &remaining);
        let h = 0.85 * greedy + 0.15 * mst;

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
