use std::collections::HashMap;

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
struct FactValue {
    variable: usize,
    value: usize,
}

struct LlmHeuristic {
    node_count: usize,
    propositional_count: usize,
    numeric_count: usize,

    depot: usize,
    location_facts: Vec<FactValue>,
    visited_facts: Vec<FactValue>,

    // Row-major directed routing-cost matrix. Node zero is not assumed to be
    // the depot; depot stores its actual task-dependent node index.
    distances: Vec<f64>,

    // Resource information is deliberately best effort. The translator may
    // fold capacity and demand expressions into unnamed derived variables.
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,
}

fn unary_atom_arg<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let normalized = name.trim().strip_prefix("Atom ").unwrap_or(name.trim());
    let prefix = format!("{}(", predicate);
    let rest = normalized.strip_prefix(&prefix)?;
    let arg = rest.strip_suffix(')')?.trim();
    if arg.is_empty() || arg.contains(',') {
        None
    } else {
        Some(arg)
    }
}

fn normalized_pne_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed).trim()
}

fn grounded_operator_tokens(name: &str) -> Vec<String> {
    name.trim()
        .trim_matches(|c| c == '(' || c == ')')
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c| c == '(' || c == ')' || c == ',')
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        // The CVRP goal identifies all customers through visited(customer)
        // facts and identifies the depot through its loc(depot) fact.
        for goal_index in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_index);
            let fact_name = task.get_fact_name(&fact);

            if let Some(customer) = unary_atom_arg(fact_name, "visited") {
                customer_names.push(customer.to_string());
            } else if let Some(place) = unary_atom_arg(fact_name, "loc") {
                depot_name = Some(place.to_string());
            }
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "missing depot location goal fact".to_string().into()
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP task has no visited-customer goal facts".to_string(),
            ));
        }

        let mut node_names: Vec<String> = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());
        let depot = 0usize;
        let node_count = node_names.len();

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        for (node, name) in node_names.iter().enumerate() {
            node_by_name.insert(name.clone(), node);
        }

        let propositional_count = task.variables().len();
        let mut optional_location_facts: Vec<Option<FactValue>> = vec![None; node_count];
        let mut optional_visited_facts: Vec<Option<FactValue>> =
            vec![None; customer_names.len()];

        // Propositional variable names are generic, so inspect every fact name
        // once during construction and retain only the compact var/value pairs.
        for variable in 0..task.variables().len() {
            for value in 0..task.variables()[variable].domain_size() {
                let fact = ExplicitFact::propositional(variable, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(place) = unary_atom_arg(fact_name, "loc") {
                    if let Some(&node) = node_by_name.get(place) {
                        optional_location_facts[node] = Some(FactValue {
                            variable,
                            value,
                        });
                    }
                }

                if let Some(customer) = unary_atom_arg(fact_name, "visited") {
                    if let Some(&node) = node_by_name.get(customer) {
                        if node != depot {
                            optional_visited_facts[node - 1] = Some(FactValue {
                                variable,
                                value,
                            });
                        }
                    }
                }
            }
        }

        let mut location_facts = Vec::with_capacity(node_count);
        for (node, fact) in optional_location_facts.into_iter().enumerate() {
            location_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                format!("missing loc fact for routing node {}", node).into()
            })?);
        }

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        for (customer, fact) in optional_visited_facts.into_iter().enumerate() {
            visited_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                format!("missing visited fact for customer index {}", customer).into()
            })?);
        }

        // Recover all routing costs from grounded operators. This avoids any
        // dependency on translator-specific travel-cost numeric-variable names.
        let mut distances = vec![f64::INFINITY; node_count * node_count];
        for node in 0..node_count {
            distances[node * node_count + node] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens = grounded_operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0].as_str() {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(&tokens[1]), node_by_name.get(&tokens[2]))
                    {
                        let slot = from * node_count + to;
                        distances[slot] = distances[slot].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = node_by_name.get(&tokens[1]) {
                        let slot = from * node_count + depot;
                        distances[slot] = distances[slot].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Every customer can be entered by visit and can return to the depot.
        // Missing entries indicate that operator parsing did not recover the
        // domain's essential routing structure.
        for from in 0..node_count {
            for to in 1..node_count {
                if from != to && !distances[from * node_count + to].is_finite() {
                    return Err(HeuristicBuildError::from(format!(
                        "missing routing cost from node {} to customer node {}",
                        from, to
                    )));
                }
            }
        }
        for customer in 1..node_count {
            if !distances[customer * node_count + depot].is_finite() {
                return Err(HeuristicBuildError::from(format!(
                    "missing return-to-depot cost for customer node {}",
                    customer
                )));
            }
        }

        // Build an exact-name map after removing only the documented PNE prefix.
        // No substring matching is used, particularly for folded derived names.
        let numeric_count = task.numeric_variables().len();
        let mut numeric_by_exact_name: HashMap<String, usize> = HashMap::new();
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            numeric_by_exact_name.insert(normalized_pne_name(variable.name()).to_string(), id);
        }

        let load_id = numeric_by_exact_name.get("load()").copied();
        let vehicles_id = numeric_by_exact_name.get("vehicles()").copied();
        let initial_numeric = task.get_initial_numeric_state_values();

        let capacity = numeric_by_exact_name
            .get("capacity()")
            .and_then(|&id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(f64::MAX);

        let max_vehicles = numeric_by_exact_name
            .get("max_vehicles()")
            .and_then(|&id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(f64::MAX);

        let mut demands = vec![0.0_f64; node_count];
        for node in 1..node_count {
            let exact_name = format!("demand({})", node_names[node]);
            if let Some(&id) = numeric_by_exact_name.get(&exact_name) {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value >= 0.0 {
                        demands[node] = *value;
                    }
                }
            }
        }

        Ok(Self {
            node_count,
            propositional_count,
            numeric_count,
            depot,
            location_facts,
            visited_facts,
            distances,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
        })
    }

    #[inline]
    fn distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from * self.node_count + to]
    }

    // A cheap spanning-tree connection estimate smooths some of the sharp
    // changes caused by nearest-neighbour greedy completion.
    fn connection_tree_estimate(&self, current: usize, unvisited: &[usize]) -> f64 {
        if unvisited.is_empty() {
            return self.distance(current, self.depot);
        }

        let count = unvisited.len();
        let mut in_tree = vec![false; count];
        let mut best = vec![f64::INFINITY; count];
        best[0] = 0.0;
        let mut tree_cost: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = f64::INFINITY;
            for i in 0..count {
                if !in_tree[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let Some(selected) = selected else {
                return f64::INFINITY;
            };

            in_tree[selected] = true;
            tree_cost += selected_cost;
            let from = unvisited[selected];

            for j in 0..count {
                if !in_tree[j] {
                    let to = unvisited[j];
                    let undirected = self.distance(from, to).min(self.distance(to, from));
                    if undirected < best[j] {
                        best[j] = undirected;
                    }
                }
            }
        }

        let mut current_connection = f64::INFINITY;
        let mut depot_connection = f64::INFINITY;
        for &customer in unvisited {
            current_connection = current_connection.min(self.distance(current, customer));
            depot_connection = depot_connection.min(self.distance(customer, self.depot));
        }

        tree_cost + current_connection + depot_connection
    }

    fn greedy_completion(
        &self,
        current: usize,
        initial_load: f64,
        unvisited: &[usize],
    ) -> f64 {
        if unvisited.is_empty() {
            return self.distance(current, self.depot);
        }

        let mut remaining = unvisited.to_vec();
        let mut position = current;
        let mut load = initial_load.max(0.0);
        let mut cost: f64 = 0.0;

        while !remaining.is_empty() {
            let mut selected: Option<usize> = None;
            let mut selected_distance = f64::INFINITY;

            for (index, &customer) in remaining.iter().enumerate() {
                let demand = self.demands[customer];
                let tolerance = 1.0e-9_f64 * self.capacity.abs().max(1.0);
                if load + demand <= self.capacity + tolerance {
                    let distance = self.distance(position, customer);
                    if distance < selected_distance {
                        selected = Some(index);
                        selected_distance = distance;
                    }
                }
            }

            if let Some(index) = selected {
                let customer = remaining.swap_remove(index);
                cost += self.distance(position, customer);
                load += self.demands[customer];
                position = customer;
            } else {
                // The current route cannot accept another remaining customer.
                // Close it and continue with a fresh vehicle from the depot.
                if position == self.depot {
                    return f64::INFINITY;
                }
                cost += self.distance(position, self.depot);
                position = self.depot;
                load = 0.0;
            }
        }

        cost + self.distance(position, self.depot)
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

        let mut propositional_values = vec![0usize; self.propositional_count];
        state.fill_state(registry, &mut propositional_values);

        let mut current: Option<usize> = None;
        for (node, fact) in self.location_facts.iter().enumerate() {
            if propositional_values[fact.variable] == fact.value {
                current = Some(node);
                break;
            }
        }
        let current = current.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "CVRP state has no recognized current location".to_string(),
            )
        })?;

        let mut unvisited: Vec<usize> = Vec::with_capacity(self.visited_facts.len());
        for (customer_index, fact) in self.visited_facts.iter().enumerate() {
            if propositional_values[fact.variable] != fact.value {
                unvisited.push(customer_index + 1);
            }
        }

        if unvisited.is_empty() {
            return Ok(self.distance(current, self.depot).max(0.0));
        }

        let mut load: f64 = 0.0;
        let mut vehicles: f64 = 0.0;
        if self.load_id.is_some() || self.vehicles_id.is_some() {
            let mut numeric_values = vec![0.0_f64; self.numeric_count];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            if let Some(id) = self.load_id {
                if let Some(value) = numeric_values.get(id) {
                    if value.is_finite() {
                        load = *value;
                    }
                }
            }
            if let Some(id) = self.vehicles_id {
                if let Some(value) = numeric_values.get(id) {
                    if value.is_finite() {
                        vehicles = *value;
                    }
                }
            }
        }

        // Necessary aggregate feasibility test. It is only activated when both
        // capacity and the vehicle limit survived translation under exact names.
        if self.capacity < f64::MAX && self.max_vehicles < f64::MAX {
            let mut remaining_demand: f64 = 0.0;
            for &customer in &unvisited {
                remaining_demand += self.demands[customer];
            }

            let residual = (self.capacity - load).max(0.0);
            let additional_vehicles = (self.max_vehicles - vehicles).max(0.0).floor();
            let available_capacity = residual + additional_vehicles * self.capacity;
            let tolerance = 1.0e-9_f64 * available_capacity.abs().max(1.0);
            if remaining_demand > available_capacity + tolerance {
                return Ok(f64::INFINITY);
            }
        }

        let greedy = self.greedy_completion(current, load, &unvisited);
        let tree = self.connection_tree_estimate(current, &unvisited);

        let h = if greedy.is_finite() && tree.is_finite() {
            // The greedy completion supplies the main, deliberately
            // non-admissible estimate. The tree term reduces sensitivity to
            // individual nearest-neighbour choices.
            0.9_f64 * greedy + 0.1_f64 * tree
        } else if greedy.is_finite() {
            greedy
        } else {
            f64::INFINITY
        };

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
