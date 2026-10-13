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

struct LlmHeuristic {
    // Node 0 is the depot; nodes 1.. are customers.
    customer_names: Vec<String>,
    visited_facts: Vec<(usize, usize)>,
    location_facts: Vec<(usize, usize)>,

    // Direct visit costs. Row/column 0 denotes the depot.
    direct_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,

    // Optional resource information. These identifiers and values are only
    // used when the translator preserves the exact original PNE names.
    load_id: Option<usize>,
    capacity: f64,
    demands: Vec<f64>,
}

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let prefix = format!("Atom {}(", predicate);
    let trimmed = name.trim();
    if trimmed.starts_with(&prefix) && trimmed.ends_with(')') {
        let begin = prefix.len();
        let end = trimmed.len() - 1;
        Some(trimmed[begin..end].trim().to_string())
    } else {
        None
    }
}

fn exact_pne_name(name: &str, core: &str) -> bool {
    let trimmed = name.trim();
    if trimmed == core {
        return true;
    }
    if let Some(rest) = trimmed.strip_prefix("PNE ") {
        return rest.trim() == core;
    }
    false
}

fn normalized_operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut depot_name: Option<String> = None;
        let mut customer_names: Vec<String> = Vec::new();

        // In this domain the goals identify both the depot and every customer.
        for goal_id in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_id);
            let fact_name = task.get_fact_name(&fact);

            if let Some(name) = atom_argument(fact_name, "loc") {
                depot_name = Some(name);
            }
            if let Some(name) = atom_argument(fact_name, "visited") {
                customer_names.push(name);
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "missing visited customer goal facts".to_string(),
            ));
        }

        let customer_count = customer_names.len();
        let node_count = customer_count + 1;

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), 0);
        for (customer_index, name) in customer_names.iter().enumerate() {
            node_by_name.insert(name.clone(), customer_index + 1);
        }

        // Resolve the translated propositional encodings once. We do not
        // assume which value of a finite-domain variable means true.
        let mut visited_facts: Vec<Option<(usize, usize)>> =
            vec![None; customer_count];
        let mut location_facts: Vec<Option<(usize, usize)>> =
            vec![None; node_count];

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = atom_argument(fact_name, "visited") {
                    if let Some(&node) = node_by_name.get(&object) {
                        if node > 0 {
                            visited_facts[node - 1] = Some((var_id, value));
                        }
                    }
                }

                if let Some(object) = atom_argument(fact_name, "loc") {
                    if let Some(&node) = node_by_name.get(&object) {
                        location_facts[node] = Some((var_id, value));
                    }
                }
            }
        }

        let visited_facts: Vec<(usize, usize)> = visited_facts
            .into_iter()
            .enumerate()
            .map(|(i, fact)| {
                fact.ok_or_else(|| -> HeuristicBuildError {
                    format!("missing translated visited fact for customer {}", i).into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let location_facts: Vec<(usize, usize)> = location_facts
            .into_iter()
            .enumerate()
            .map(|(i, fact)| {
                fact.ok_or_else(|| -> HeuristicBuildError {
                    format!("missing translated location fact for node {}", i).into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Reconstruct the routing graph from grounded operators. This avoids
        // depending on travel-cost PNE names, which may be folded away.
        let mut direct_cost = vec![vec![f64::INFINITY; node_count]; node_count];
        let mut return_cost = vec![f64::INFINITY; node_count];
        for i in 0..node_count {
            direct_cost[i][i] = 0.0;
        }
        return_cost[0] = 0.0;

        for op in task.get_operators() {
            let tokens = normalized_operator_tokens(op.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(tokens[1]), node_by_name.get(tokens[2]))
                    {
                        if cost < direct_cost[from][to] {
                            direct_cost[from][to] = cost;
                        }
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = node_by_name.get(tokens[1]) {
                        if cost < return_cost[from] {
                            return_cost[from] = cost;
                        }
                    }
                }
                _ => {}
            }
        }

        // The CVRP grounding should contain a direct visit between every
        // distinct relevant pair, plus every customer-to-depot return.
        for customer in 1..node_count {
            if !direct_cost[0][customer].is_finite() {
                return Err(HeuristicBuildError::from(format!(
                    "missing depot-to-customer routing cost for node {}",
                    customer
                )));
            }
            if !return_cost[customer].is_finite() {
                return Err(HeuristicBuildError::from(format!(
                    "missing customer-to-depot routing cost for node {}",
                    customer
                )));
            }
            direct_cost[customer][0] = return_cost[customer];
        }

        for from in 1..node_count {
            for to in 1..node_count {
                if from != to && !direct_cost[from][to].is_finite() {
                    return Err(HeuristicBuildError::from(format!(
                        "missing direct routing cost from node {} to node {}",
                        from, to
                    )));
                }
            }
        }

        // Resolve optional numeric resources using exact full PNE names only.
        let mut load_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; customer_count];

        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if exact_pne_name(name, "load()") {
                load_id = Some(numeric_id);
            }
            if exact_pne_name(name, "capacity()") {
                capacity_id = Some(numeric_id);
            }

            for (customer_index, customer_name) in customer_names.iter().enumerate() {
                let exact_demand = format!("demand({})", customer_name);
                if exact_pne_name(name, &exact_demand) {
                    demand_ids[customer_index] = Some(numeric_id);
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(f64::MAX);

        let mut demands: Vec<f64> = vec![0.0; customer_count];
        for customer_index in 0..customer_count {
            if let Some(id) = demand_ids[customer_index] {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value >= 0.0 {
                        demands[customer_index] = *value;
                    }
                }
            }
        }

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts,
            direct_cost,
            return_cost,
            load_id,
            capacity,
            demands,
        })
    }

    fn symmetric_edge_cost(&self, a: usize, b: usize) -> f64 {
        self.direct_cost[a][b].min(self.direct_cost[b][a])
    }

    fn mst_bound(&self, nodes: &[usize]) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let count = nodes.len();
        let mut in_tree = vec![false; count];
        let mut best = vec![f64::INFINITY; count];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;
            for i in 0..count {
                if !in_tree[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let Some(u) = selected else {
                return total;
            };
            in_tree[u] = true;
            total += selected_cost;

            for v in 0..count {
                if !in_tree[v] {
                    let edge = self.symmetric_edge_cost(nodes[u], nodes[v]);
                    if edge < best[v] {
                        best[v] = edge;
                    }
                }
            }
        }

        total
    }

    fn greedy_completion(
        &self,
        start: usize,
        initial_load: f64,
        unvisited: &[bool],
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|value| **value).count();
        let mut current = start;
        let mut load = if start == 0 { 0.0 } else { initial_load.max(0.0) };
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;

            // Prefer a cheap direct continuation that still fits in the
            // current vehicle when exact capacity information is available.
            for customer_index in 0..remaining.len() {
                if !remaining[customer_index] {
                    continue;
                }

                let node = customer_index + 1;
                let demand = self.demands[customer_index];
                let fits = self.capacity == f64::MAX
                    || load + demand <= self.capacity + 1.0e-9;

                if fits {
                    let edge = self.direct_cost[current][node];
                    if edge < selected_cost {
                        selected_cost = edge;
                        selected = Some(customer_index);
                    }
                }
            }

            if let Some(customer_index) = selected {
                let node = customer_index + 1;
                total += self.direct_cost[current][node];
                load += self.demands[customer_index];
                current = node;
                remaining[customer_index] = false;
                remaining_count -= 1;
                continue;
            }

            // Nothing fits. The domain's visit-via-depot action closes the
            // current route and immediately starts another at a customer.
            let mut restart: Option<usize> = None;
            let mut restart_cost = f64::INFINITY;
            for customer_index in 0..remaining.len() {
                if remaining[customer_index] {
                    let node = customer_index + 1;
                    let cost = self.direct_cost[0][node];
                    if cost < restart_cost {
                        restart_cost = cost;
                        restart = Some(customer_index);
                    }
                }
            }

            let Some(customer_index) = restart else {
                break;
            };
            let node = customer_index + 1;
            if current != 0 {
                total += self.return_cost[current];
            }
            total += self.direct_cost[0][node];
            load = self.demands[customer_index];
            current = node;
            remaining[customer_index] = false;
            remaining_count -= 1;
        }

        if current != 0 {
            total += self.return_cost[current];
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

        let mut propositional_values = vec![0usize; task.variables().len()];
        state.fill_state(registry, &mut propositional_values);

        let mut current_node = 0usize;
        for (node, &(var_id, value)) in self.location_facts.iter().enumerate() {
            if propositional_values[var_id] == value {
                current_node = node;
                break;
            }
        }

        let mut unvisited = vec![false; self.customer_names.len()];
        let mut remaining_count = 0usize;
        for (customer_index, &(var_id, value)) in
            self.visited_facts.iter().enumerate()
        {
            if propositional_values[var_id] != value {
                unvisited[customer_index] = true;
                remaining_count += 1;
            }
        }

        // All customers are served but the vehicle has not yet returned.
        if remaining_count == 0 {
            return Ok(if current_node == 0 {
                0.0
            } else {
                self.return_cost[current_node]
            });
        }

        let mut current_load: f64 = 0.0;
        if let Some(load_id) = self.load_id {
            let mut numeric_values = vec![0.0_f64; task.numeric_variables().len()];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
            if let Some(value) = numeric_values.get(load_id) {
                if value.is_finite() {
                    current_load = *value;
                }
            }
        }

        // Prim MST over depot, current position, and unvisited customers.
        // This captures the global spatial structure and avoids the severe
        // myopia of a nearest-neighbour estimate alone.
        let mut mst_nodes: Vec<usize> = Vec::with_capacity(remaining_count + 2);
        mst_nodes.push(0);
        if current_node != 0 {
            mst_nodes.push(current_node);
        }
        for customer_index in 0..unvisited.len() {
            if unvisited[customer_index] {
                mst_nodes.push(customer_index + 1);
            }
        }
        let mst = self.mst_bound(&mst_nodes);

        // Construct a cheap feasible-style completion, inserting depot route
        // breaks whenever known capacity prevents another direct visit.
        let greedy = self.greedy_completion(
            current_node,
            current_load,
            &unvisited,
        );

        // The completion estimate is more informative for CVRP route choices;
        // the MST component stabilizes it against nearest-neighbour mistakes.
        let h = (0.8_f64 * greedy + 0.2_f64 * mst).max(0.0);
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
