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

fn atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let atom = name.trim().strip_prefix("Atom ")?;
    let rest = atom.strip_prefix(predicate)?.strip_prefix('(')?;
    rest.strip_suffix(')')
}

fn normalized_pne_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed)
}

fn exact_zero_arity_pne(name: &str, wanted: &str) -> bool {
    normalized_pne_name(name) == wanted
}

fn exact_unary_pne_argument<'a>(name: &'a str, function: &str) -> Option<&'a str> {
    let normalized = normalized_pne_name(name);
    let rest = normalized.strip_prefix(function)?.strip_prefix('(')?;
    rest.strip_suffix(')')
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<ExplicitFact>,
    customer_loc_facts: Vec<ExplicitFact>,
    depot_loc_fact: ExplicitFact,

    // Node zero is the depot; customer i is node i + 1.
    distances: Vec<Vec<f64>>,
    fallback_distance: f64,

    numeric_count: usize,
    load_id: Option<usize>,
    capacity: Option<f64>,
    demands: Vec<Option<f64>>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customers: Vec<(String, ExplicitFact)> = Vec::new();
        let mut locations: Vec<(String, ExplicitFact)> = Vec::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = atom_argument(fact_name, "visited") {
                    if !customers.iter().any(|(name, _)| name == object) {
                        customers.push((object.to_string(), fact));
                    }
                }

                if let Some(object) = atom_argument(fact_name, "loc") {
                    if !locations.iter().any(|(name, _)| name == object) {
                        locations.push((object.to_string(), fact));
                    }
                }
            }
        }

        let customer_names: Vec<String> = customers
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let visited_facts: Vec<ExplicitFact> = customers
            .iter()
            .map(|(_, fact)| fact.clone())
            .collect();

        let depot_entry = locations
            .iter()
            .find(|(name, _)| !customer_names.iter().any(|customer| customer == name))
            .ok_or_else(|| -> HeuristicBuildError {
                "missing depot loc fact".to_string().into()
            })?;
        let depot_name = depot_entry.0.clone();
        let depot_loc_fact = depot_entry.1.clone();

        let mut customer_loc_facts = Vec::with_capacity(customer_names.len());
        for customer in &customer_names {
            let fact = locations
                .iter()
                .find(|(name, _)| name == customer)
                .map(|(_, fact)| fact.clone())
                .ok_or_else(|| -> HeuristicBuildError {
                    format!("missing loc fact for customer {customer}").into()
                })?;
            customer_loc_facts.push(fact);
        }

        let node_count = customer_names.len() + 1;
        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for i in 0..node_count {
            distances[i][i] = 0.0;
        }

        let node_index = |object: &str| -> Option<usize> {
            if object == depot_name {
                Some(0)
            } else {
                customer_names
                    .iter()
                    .position(|name| name == object)
                    .map(|index| index + 1)
            }
        };

        for op in task.get_operators() {
            let tokens = operator_tokens(op.name());
            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(from_id), Some(to_id)) = (node_index(from), node_index(to)) {
                        distances[from_id][to_id] = distances[from_id][to_id].min(cost);
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(from_id) = node_index(from) {
                        distances[from_id][0] = distances[from_id][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Preserve all directly reconstructed arcs. If a translator omitted one
        // direction, use its reverse as a conservative task-independent fallback.
        for i in 0..node_count {
            for j in 0..node_count {
                if !distances[i][j].is_finite() && distances[j][i].is_finite() {
                    distances[i][j] = distances[j][i];
                }
            }
        }

        let mut largest_finite: f64 = 1.0;
        for row in &distances {
            for &distance in row {
                if distance.is_finite() {
                    largest_finite = largest_finite.max(distance);
                }
            }
        }
        let fallback_distance = largest_finite * 2.0;

        let numeric_count = task.numeric_variables().len();
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut capacity_id = None;
        let mut demand_ids = vec![None; customer_names.len()];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if exact_zero_arity_pne(name, "load()") {
                load_id = Some(id);
            } else if exact_zero_arity_pne(name, "capacity()") {
                capacity_id = Some(id);
            } else if let Some(customer) = exact_unary_pne_argument(name, "demand") {
                if let Some(index) = customer_names.iter().position(|name| name == customer) {
                    demand_ids[index] = Some(id);
                }
            }
        }

        let capacity = capacity_id.and_then(|id| {
            initial_numeric
                .get(id)
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
        });

        let demands = demand_ids
            .into_iter()
            .map(|id| {
                id.and_then(|numeric_id| {
                    initial_numeric
                        .get(numeric_id)
                        .copied()
                        .filter(|value| value.is_finite() && *value >= 0.0)
                })
            })
            .collect();

        Ok(Self {
            customer_names,
            visited_facts,
            customer_loc_facts,
            depot_loc_fact,
            distances,
            fallback_distance,
            numeric_count,
            load_id,
            capacity,
            demands,
        })
    }

    fn edge_weight(&self, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }

        let ab = self.distances[a][b];
        let ba = self.distances[b][a];
        if ab.is_finite() && ba.is_finite() {
            ab.min(ba)
        } else if ab.is_finite() {
            ab
        } else if ba.is_finite() {
            ba
        } else {
            self.fallback_distance
        }
    }

    fn directed_distance(&self, a: usize, b: usize) -> f64 {
        let direct = self.distances[a][b];
        if direct.is_finite() {
            direct
        } else {
            self.edge_weight(a, b)
        }
    }

    fn mst_cost(&self, active: &[usize]) -> f64 {
        if active.len() <= 1 {
            return 0.0;
        }

        let n = active.len();
        let mut used = vec![false; n];
        let mut best = vec![f64::INFINITY; n];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..n {
            let mut selected = None;
            let mut selected_cost = f64::INFINITY;
            for i in 0..n {
                if !used[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let Some(u) = selected else {
                return total + self.fallback_distance;
            };
            used[u] = true;
            total += selected_cost;

            for v in 0..n {
                if !used[v] {
                    let edge = self.edge_weight(active[u], active[v]);
                    if edge < best[v] {
                        best[v] = edge;
                    }
                }
            }
        }

        total
    }

    fn degree_bound(
        &self,
        active: &[usize],
        current_node: usize,
        depot_degree: usize,
    ) -> f64 {
        let mut degree_sum: f64 = 0.0;

        // Every unvisited customer must eventually have degree two. If the
        // vehicle is at a customer, that current endpoint needs one future edge.
        for &node in active {
            if node == 0 {
                continue;
            }

            let required = if node == current_node && current_node != 0 {
                1
            } else {
                2
            };

            let mut candidates = Vec::with_capacity(active.len() + 1);
            for &other in active {
                if other == node {
                    continue;
                }
                let weight = self.edge_weight(node, other);
                candidates.push(weight);
                // A singleton route may use the customer-depot edge twice.
                if other == 0 {
                    candidates.push(weight);
                }
            }
            candidates.sort_by(|a, b| a.total_cmp(b));

            for index in 0..required {
                degree_sum += candidates
                    .get(index)
                    .copied()
                    .unwrap_or(self.fallback_distance);
            }
        }

        let mut depot_candidates = Vec::with_capacity(active.len() * 2);
        for &node in active {
            if node == 0 {
                continue;
            }
            let weight = self.edge_weight(0, node);
            depot_candidates.push(weight);
            if node != current_node || current_node == 0 {
                depot_candidates.push(weight);
            }
        }
        depot_candidates.sort_by(|a, b| a.total_cmp(b));

        for index in 0..depot_degree {
            degree_sum += depot_candidates
                .get(index)
                .copied()
                .unwrap_or(self.fallback_distance);
        }

        0.5 * degree_sum
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

        let mut current_node = 0usize;
        if !self.depot_loc_fact.is_hold(view) {
            for (index, fact) in self.customer_loc_facts.iter().enumerate() {
                if fact.is_hold(registry.view(state)) {
                    current_node = index + 1;
                    break;
                }
            }
        }

        let mut unvisited = Vec::with_capacity(self.customer_names.len());
        for (index, fact) in self.visited_facts.iter().enumerate() {
            if !fact.is_hold(registry.view(state)) {
                unvisited.push(index);
            }
        }

        if unvisited.is_empty() {
            return Ok(self.directed_distance(current_node, 0).max(0.0));
        }

        let mut active = Vec::with_capacity(unvisited.len() + 2);
        active.push(0);
        if current_node != 0 {
            active.push(current_node);
        }
        for &customer in &unvisited {
            active.push(customer + 1);
        }

        let mst = self.mst_cost(&active);

        // Without recoverable resource constants, use the one-route relaxation.
        let mut depot_degree = if current_node == 0 { 2usize } else { 1usize };

        if let (Some(capacity), Some(load_id)) = (self.capacity, self.load_id) {
            let all_demands_available = unvisited
                .iter()
                .all(|&customer| self.demands[customer].is_some());

            if all_demands_available {
                let mut numeric_values = vec![0.0; self.numeric_count];
                registry
                    .view(state)
                    .fill_numeric(&mut numeric_values)
                    .map_err(|err| {
                        EvaluationError::ComputationFailed(format!("{err:?}"))
                    })?;

                let remaining_demand: f64 = unvisited
                    .iter()
                    .map(|&customer| self.demands[customer].unwrap_or(0.0))
                    .sum();
                let load = numeric_values
                    .get(load_id)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0);

                if current_node == 0 {
                    let routes = (remaining_demand / capacity).ceil().max(1.0) as usize;
                    depot_degree = routes.saturating_mul(2);
                } else {
                    let residual_capacity = (capacity - load).max(0.0);
                    let demand_after_current_route =
                        (remaining_demand - residual_capacity).max(0.0);
                    let additional_routes =
                        (demand_after_current_route / capacity).ceil() as usize;
                    depot_degree = 1usize.saturating_add(additional_routes.saturating_mul(2));
                }
            }
        }

        let degree = self.degree_bound(&active, current_node, depot_degree);
        let h = mst.max(degree).max(0.0);

        Ok(if h.is_finite() { h } else { self.fallback_distance });
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
