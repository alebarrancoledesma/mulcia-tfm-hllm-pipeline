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

#[derive(Clone, Copy)]
struct FactKey {
    var: usize,
    value: usize,
}

struct LlmHeuristic {
    visited_facts: Vec<FactKey>,
    loc_facts: Vec<FactKey>,
    distances: Vec<Vec<f64>>,
    capacity_id: Option<usize>,
    load_id: Option<usize>,
    demand_ids: Vec<Option<usize>>,
}

fn parse_atom_argument(text: &str, predicate: &str) -> Option<String> {
    let text = text.trim();
    let prefix = format!("Atom {}(", predicate);
    let rest = text.strip_prefix(&prefix)?;
    let argument = rest.strip_suffix(')')?;
    Some(argument.trim().to_string())
}

fn normalized_numeric_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed)
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Customers are discovered from the visited goals. The depot is the
        // object occurring in the remaining loc goal.
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        for goal_id in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_id);
            let fact_name = task.get_fact_name(&fact).to_string();

            if let Some(customer) = parse_atom_argument(&fact_name, "visited") {
                customer_names.push(customer);
            } else if let Some(place) = parse_atom_argument(&fact_name, "loc") {
                depot_name = Some(place);
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot loc goal fact".to_string().into()
        })?;

        let customer_count = customer_names.len();
        let node_count = customer_count + 1;

        // Routing-node index 0 is the dynamically discovered depot. Customer
        // indices are their positions in the goal-derived customer vector + 1.
        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), 0);
        for (customer_id, customer) in customer_names.iter().enumerate() {
            node_by_name.insert(customer.clone(), customer_id + 1);
        }

        let mut visited_facts: Vec<Option<FactKey>> = vec![None; customer_count];
        let mut loc_facts: Vec<Option<FactKey>> = vec![None; node_count];

        // Propositional variable names are generic, so semantic facts are
        // resolved once by enumerating their finite-domain values.
        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(place) = parse_atom_argument(&fact_name, "loc") {
                    if let Some(&node) = node_by_name.get(&place) {
                        loc_facts[node] = Some(FactKey { var: var_id, value });
                    }
                }

                if let Some(customer) = parse_atom_argument(&fact_name, "visited") {
                    if let Some(&node) = node_by_name.get(&customer) {
                        if node > 0 {
                            visited_facts[node - 1] =
                                Some(FactKey { var: var_id, value });
                        }
                    }
                }
            }
        }

        let mut final_visited_facts: Vec<FactKey> = Vec::with_capacity(customer_count);
        for fact in visited_facts {
            final_visited_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                "missing visited fact for a customer goal".to_string().into()
            })?);
        }

        let mut final_loc_facts: Vec<FactKey> = Vec::with_capacity(node_count);
        for fact in loc_facts {
            final_loc_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                "missing loc fact for a routing place".to_string().into()
            })?);
        }

        // Resolve optional numeric resource variables only by exact normalized
        // names. Their absence is expected when the translator folds constants
        // into derived expressions.
        let mut capacity_id: Option<usize> = None;
        let mut load_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; customer_count];

        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            let name = normalized_numeric_name(variable.name());

            if name == "capacity()" {
                capacity_id = Some(numeric_id);
            } else if name == "load()" {
                load_id = Some(numeric_id);
            } else {
                for (customer_id, customer) in customer_names.iter().enumerate() {
                    let expected = format!("demand({})", customer);
                    if name == expected {
                        demand_ids[customer_id] = Some(numeric_id);
                        break;
                    }
                }
            }
        }

        // Reconstruct routing costs from grounded operators rather than from
        // translator-dependent travel-cost numeric-variable names.
        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for i in 0..node_count {
            distances[i][i] = 0.0;
        }

        let mut largest_cost: f64 = 0.0;

        for op in task.get_operators() {
            let cleaned_name = op.name().trim().trim_matches(|c| c == '(' || c == ')');
            let tokens: Vec<&str> = cleaned_name.split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, op);

            if !cost.is_finite() || cost < 0.0 {
                continue;
            }
            largest_cost = largest_cost.max(cost);

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_by_name.get(*from), node_by_name.get(*to))
                    {
                        distances[from_id][to_id] =
                            distances[from_id][to_id].min(cost);
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = node_by_name.get(*from) {
                        distances[from_id][0] = distances[from_id][0].min(cost);
                    }
                }
                ["visit-via-depot", from, to] => {
                    // This is a valid relaxed transition and is also a useful
                    // fallback if a direct visit arc was omitted by grounding.
                    if let (Some(&from_id), Some(&to_id)) =
                        (node_by_name.get(*from), node_by_name.get(*to))
                    {
                        distances[from_id][to_id] =
                            distances[from_id][to_id].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Complete any accidentally sparse cost table through shortest paths.
        // Normally the grounded visit operators already provide a complete
        // directed routing graph.
        for k in 0..node_count {
            for i in 0..node_count {
                if !distances[i][k].is_finite() {
                    continue;
                }
                for j in 0..node_count {
                    let through = distances[i][k] + distances[k][j];
                    if through < distances[i][j] {
                        distances[i][j] = through;
                    }
                }
            }
        }

        let fallback = if largest_cost > 0.0 {
            largest_cost * (node_count as f64 + 1.0)
        } else {
            1.0
        };
        for row in &mut distances {
            for distance in row {
                if !distance.is_finite() {
                    *distance = fallback;
                }
            }
        }

        Ok(Self {
            visited_facts: final_visited_facts,
            loc_facts: final_loc_facts,
            distances,
            capacity_id,
            load_id,
            demand_ids,
        })
    }

    fn undirected_cost(&self, a: usize, b: usize) -> f64 {
        self.distances[a][b].min(self.distances[b][a])
    }

    fn mst_cost(&self, nodes: &[usize]) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let n = nodes.len();
        let mut used = vec![false; n];
        let mut best = vec![f64::INFINITY; n];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..n {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;

            for i in 0..n {
                if !used[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let Some(u) = selected else {
                break;
            };
            used[u] = true;
            total += selected_cost;

            for v in 0..n {
                if !used[v] {
                    let edge = self.undirected_cost(nodes[u], nodes[v]);
                    if edge < best[v] {
                        best[v] = edge;
                    }
                }
            }
        }

        total
    }

    fn degree_bound(&self, nodes: &[usize], current: usize) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let current_is_depot = current == 0;
        let mut degree_sum: f64 = 0.0;

        for &node in nodes {
            let required_degree = if current_is_depot {
                2
            } else if node == current || node == 0 {
                1
            } else {
                2
            };

            let mut first = f64::INFINITY;
            let mut second = f64::INFINITY;
            for &other in nodes {
                if other == node {
                    continue;
                }
                let edge = self.undirected_cost(node, other);
                if edge < first {
                    second = first;
                    first = edge;
                } else if edge < second {
                    second = edge;
                }
            }

            if required_degree == 1 {
                degree_sum += first;
            } else if second.is_finite() {
                degree_sum += first + second;
            } else {
                // With only two distinct nodes, the route traverses the same
                // undirected connection in both directions.
                degree_sum += 2.0 * first;
            }
        }

        0.5 * degree_sum
    }

    fn nearest_neighbor_completion(&self, current: usize, unvisited: &[usize]) -> f64 {
        let mut remaining = vec![false; self.distances.len()];
        for &node in unvisited {
            remaining[node] = true;
        }

        let mut position = current;
        let mut total: f64 = 0.0;

        for _ in 0..unvisited.len() {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;

            for &candidate in unvisited {
                if remaining[candidate]
                    && self.distances[position][candidate] < selected_cost
                {
                    selected = Some(candidate);
                    selected_cost = self.distances[position][candidate];
                }
            }

            if let Some(next) = selected {
                total += selected_cost;
                remaining[next] = false;
                position = next;
            }
        }

        total + self.distances[position][0]
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

        let mut current: Option<usize> = None;
        for (node, fact) in self.loc_facts.iter().enumerate() {
            if propositional_values[fact.var] == fact.value {
                current = Some(node);
                break;
            }
        }

        let current = current.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "no recognized current CVRP location".to_string(),
            )
        })?;

        let mut unvisited: Vec<usize> = Vec::new();
        for (customer_id, fact) in self.visited_facts.iter().enumerate() {
            if propositional_values[fact.var] != fact.value {
                unvisited.push(customer_id + 1);
            }
        }

        if unvisited.is_empty() {
            return Ok(self.distances[current][0].max(0.0));
        }

        // Build the unique node set containing the two path endpoints and all
        // customers still requiring service.
        let mut active_nodes: Vec<usize> = Vec::with_capacity(unvisited.len() + 2);
        active_nodes.push(current);
        if current != 0 {
            active_nodes.push(0);
        }
        active_nodes.extend(unvisited.iter().copied());

        let mst = self.mst_cost(&active_nodes);
        let degree = self.degree_bound(&active_nodes, current);
        let routing_lower_bound = mst.max(degree);
        let greedy_completion = self.nearest_neighbor_completion(current, &unvisited);

        // The greedy route is informative but can overestimate due to locally
        // poor choices. Discounting it and taking the maximum with structural
        // bounds gives a stable, strongly discriminating routing estimate.
        let mut h = routing_lower_bound.max(0.9 * greedy_completion);

        // Optional capacity correction. It is used only if capacity, load, and
        // every relevant demand survived translation under exact names.
        let resources_available = self.capacity_id.is_some()
            && self.load_id.is_some()
            && unvisited
                .iter()
                .all(|&node| self.demand_ids[node - 1].is_some());

        if resources_available {
            let mut numeric_values =
                vec![0.0_f64; task.numeric_variables().len()];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            let capacity = numeric_values[self.capacity_id.unwrap()];
            let load = numeric_values[self.load_id.unwrap()];

            if capacity.is_finite() && capacity > 0.0 && load.is_finite() {
                let mut remaining_demand: f64 = 0.0;
                for &node in &unvisited {
                    let demand_id = self.demand_ids[node - 1].unwrap();
                    let demand = numeric_values[demand_id];
                    if demand.is_finite() && demand > 0.0 {
                        remaining_demand += demand;
                    }
                }

                let residual_capacity = if current == 0 {
                    capacity
                } else {
                    (capacity - load).max(0.0)
                };

                let extra_demand = (remaining_demand - residual_capacity).max(0.0);
                let additional_routes = (extra_demand / capacity).ceil();

                if additional_routes > 0.0 {
                    let mut minimum_departure = f64::INFINITY;
                    let mut minimum_return = f64::INFINITY;
                    for &node in &unvisited {
                        minimum_departure =
                            minimum_departure.min(self.distances[0][node]);
                        minimum_return =
                            minimum_return.min(self.distances[node][0]);
                    }

                    let radial_penalty =
                        additional_routes * (minimum_departure + minimum_return);
                    h = h.max(routing_lower_bound + radial_penalty);
                    h = h.max(0.9 * greedy_completion + radial_penalty);
                }
            }
        }

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
