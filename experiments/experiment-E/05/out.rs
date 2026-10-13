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

use std::cmp::Ordering;
use std::collections::HashMap;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

#[derive(Clone, Copy)]
struct FactKey {
    var_id: usize,
    value: usize,
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<FactKey>,
    location_facts: Vec<FactKey>,
    depot_location_fact: FactKey,

    // Customer indices are 0..n and the depot has index n.
    distance: Vec<Vec<f64>>,
    depot: usize,

    // Optional resource information. Static capacity and demands are read only
    // when their exact, non-derived PNE names survive translation.
    load_id: Option<usize>,
    capacity: Option<f64>,
    demands: Vec<Option<f64>>,
    numeric_variable_count: usize,
}

fn atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let prefix = format!("Atom {}(", predicate);
    let trimmed = name.trim();
    if !trimmed.starts_with(&prefix) || !trimmed.ends_with(')') {
        return None;
    }

    let argument = &trimmed[prefix.len()..trimmed.len() - 1];
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.trim())
    }
}

fn exact_zero_arity_pne(name: &str, fluent: &str) -> bool {
    let trimmed = name.trim();
    trimmed == format!("{}()", fluent)
        || trimmed == format!("PNE {}()", fluent)
}

fn exact_unary_pne(name: &str, fluent: &str, object: &str) -> bool {
    let trimmed = name.trim();
    trimmed == format!("{}({})", fluent, object)
        || trimmed == format!("PNE {}({})", fluent, object)
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_fact_map: HashMap<String, FactKey> = HashMap::new();

        // The true value of a translated Boolean variable is not assumed to be
        // zero. Identify it from the semantic fact name instead.
        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(customer) = atom_argument(fact_name, "visited") {
                    customer_fact_map.insert(
                        customer.to_string(),
                        FactKey { var_id, value },
                    );
                }
            }
        }

        if customer_fact_map.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not identify any visited(customer) facts".to_string(),
            ));
        }

        let mut customer_names: Vec<String> =
            customer_fact_map.keys().cloned().collect();
        customer_names.sort();

        let customer_index: HashMap<String, usize> = customer_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();

        let visited_facts: Vec<FactKey> = customer_names
            .iter()
            .map(|name| customer_fact_map[name])
            .collect();

        let n = customer_names.len();
        let depot = n;
        let mut location_facts: Vec<Option<FactKey>> = vec![None; n];
        let mut depot_location_fact: Option<FactKey> = None;

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(place) = atom_argument(fact_name, "loc") {
                    let key = FactKey { var_id, value };
                    if let Some(&index) = customer_index.get(place) {
                        location_facts[index] = Some(key);
                    } else if depot_location_fact.is_none() {
                        // In this CVRP domain the only non-customer place is the
                        // depot. This avoids hardcoding its object name.
                        depot_location_fact = Some(key);
                    }
                }
            }
        }

        let depot_location_fact = depot_location_fact.ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 could not identify the depot location fact"
                    .to_string()
                    .into()
            },
        )?;

        let mut concrete_location_facts = Vec::with_capacity(n);
        for (index, fact) in location_facts.into_iter().enumerate() {
            concrete_location_facts.push(fact.ok_or_else(
                || -> HeuristicBuildError {
                    format!(
                        "llm_001 could not identify the location fact for {}",
                        customer_names[index]
                    )
                    .into()
                },
            )?);
        }

        let mut distance = vec![vec![f64::INFINITY; n + 1]; n + 1];
        for (index, row) in distance.iter_mut().enumerate() {
            row[index] = 0.0;
        }

        // Reconstruct all useful routing costs from grounded operators. This is
        // independent of whether travel-cost PNEs were retained or folded.
        for operator in task.get_operators() {
            let parts: Vec<&str> = operator.name().split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            if parts.len() == 3 && parts[0] == "visit" {
                let from = customer_index
                    .get(parts[1])
                    .copied()
                    .unwrap_or(depot);
                if let Some(&to) = customer_index.get(parts[2]) {
                    distance[from][to] = distance[from][to].min(cost);
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&from) = customer_index.get(parts[1]) {
                    distance[from][depot] = distance[from][depot].min(cost);
                }
            }
        }

        // Exact-name-only lookup for optional resource PNEs. In particular, no
        // synthesized derived expression is matched by substring.
        let initial_numeric = task.get_initial_numeric_state_values();
        let numeric_variables = task.numeric_variables();
        let mut load_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut demands: Vec<Option<f64>> = vec![None; n];

        for (numeric_id, variable) in numeric_variables.iter().enumerate() {
            let name = variable.name();
            if exact_zero_arity_pne(name, "load") {
                load_id = Some(numeric_id);
            } else if exact_zero_arity_pne(name, "capacity") {
                let value = initial_numeric[numeric_id];
                if value.is_finite() && value > 0.0 {
                    capacity = Some(value);
                }
            } else {
                for (customer, customer_name) in customer_names.iter().enumerate() {
                    if exact_unary_pne(name, "demand", customer_name) {
                        let value = initial_numeric[numeric_id];
                        if value.is_finite() && value >= 0.0 {
                            demands[customer] = Some(value);
                        }
                        break;
                    }
                }
            }
        }

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts: concrete_location_facts,
            depot_location_fact,
            distance,
            depot,
            load_id,
            capacity,
            demands,
            numeric_variable_count: numeric_variables.len(),
        })
    }

    fn edge(&self, from: usize, to: usize) -> f64 {
        self.distance[from][to]
    }

    fn undirected_edge(&self, a: usize, b: usize) -> f64 {
        self.edge(a, b).min(self.edge(b, a))
    }

    fn mst_cost(&self, nodes: &[usize]) -> f64 {
        if nodes.len() <= 1 {
            return 0.0;
        }

        let count = nodes.len();
        let mut used = vec![false; count];
        let mut best = vec![f64::INFINITY; count];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            for index in 0..count {
                if !used[index]
                    && selected.map_or(true, |old| best[index] < best[old])
                {
                    selected = Some(index);
                }
            }

            let Some(selected) = selected else {
                return 0.0;
            };
            if !best[selected].is_finite() {
                return 0.0;
            }

            used[selected] = true;
            total += best[selected];

            for other in 0..count {
                if !used[other] {
                    let edge = self.undirected_edge(nodes[selected], nodes[other]);
                    if edge < best[other] {
                        best[other] = edge;
                    }
                }
            }
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

        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut propositional_values = Vec::new();
        state.fill_state(registry, &mut propositional_values);

        let holds = |fact: FactKey| -> bool {
            propositional_values
                .get(fact.var_id)
                .map_or(false, |&value| value == fact.value)
        };

        let current = if holds(self.depot_location_fact) {
            self.depot
        } else {
            self.location_facts
                .iter()
                .position(|&fact| holds(fact))
                .ok_or_else(|| {
                    EvaluationError::ComputationFailed(
                        "llm_001 could not identify the current vehicle location"
                            .to_string(),
                    )
                })?
        };

        let mut unvisited = Vec::with_capacity(self.customer_names.len());
        for customer in 0..self.customer_names.len() {
            if !holds(self.visited_facts[customer]) {
                unvisited.push(customer);
            }
        }

        if unvisited.is_empty() {
            let return_cost = self.edge(current, self.depot);
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                0.0
            });
        }

        // Global connectivity term: an MST over the current position, all
        // unvisited customers, and the depot. It captures clusters and remote
        // groups without committing to a customer order.
        let mut mst_nodes = Vec::with_capacity(unvisited.len() + 2);
        mst_nodes.push(current);
        mst_nodes.extend(unvisited.iter().copied());
        if current != self.depot {
            mst_nodes.push(self.depot);
        }
        let mst = self.mst_cost(&mst_nodes);

        // Construct a fast directed nearest-neighbour completion from the
        // current position through all remaining customers and back to depot.
        let mut remaining = unvisited.clone();
        let mut route = Vec::with_capacity(remaining.len() + 2);
        route.push(current);
        let mut cursor = current;
        let mut route_cost: f64 = 0.0;

        while !remaining.is_empty() {
            let mut best_position = 0usize;
            let mut best_cost = f64::INFINITY;
            for (position, &customer) in remaining.iter().enumerate() {
                let cost = self.edge(cursor, customer);
                if cost < best_cost {
                    best_cost = cost;
                    best_position = position;
                }
            }

            if !best_cost.is_finite() {
                route_cost = mst;
                break;
            }

            let next = remaining.swap_remove(best_position);
            route_cost += best_cost;
            route.push(next);
            cursor = next;
        }

        if route_cost != mst || remaining.is_empty() {
            let return_cost = self.edge(cursor, self.depot);
            if return_cost.is_finite() {
                route_cost += return_cost;
                route.push(self.depot);
            } else {
                route_cost = mst;
            }
        }

        // If exact capacity, demand, and load variables are available, estimate
        // how many extra depot transitions are forced by aggregate remaining
        // demand. Add the cheapest break costs along the constructed route.
        if let (Some(capacity), Some(load_id)) = (self.capacity, self.load_id) {
            if self.demands.iter().all(Option::is_some) {
                let mut numeric_values = vec![0.0; self.numeric_variable_count];
                registry
                    .view(state)
                    .fill_numeric(&mut numeric_values)
                    .map_err(|err| {
                        EvaluationError::ComputationFailed(format!("{err:?}"))
                    })?;

                let load = numeric_values
                    .get(load_id)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0);
                let residual_capacity = (capacity - load).max(0.0);
                let remaining_demand: f64 = unvisited
                    .iter()
                    .map(|&customer| self.demands[customer].unwrap_or(0.0))
                    .sum();

                let excess = (remaining_demand - residual_capacity).max(0.0);
                let required_breaks = if excess <= 0.0 {
                    0usize
                } else {
                    (excess / capacity).ceil() as usize
                };

                if required_breaks > 0 && route.len() >= 3 {
                    let mut break_deltas = Vec::with_capacity(route.len() - 2);
                    for position in 0..route.len() - 2 {
                        let from = route[position];
                        let to = route[position + 1];
                        if to == self.depot {
                            continue;
                        }

                        let direct = self.edge(from, to);
                        let to_depot = self.edge(from, self.depot);
                        let from_depot = self.edge(self.depot, to);
                        if direct.is_finite()
                            && to_depot.is_finite()
                            && from_depot.is_finite()
                        {
                            break_deltas.push(to_depot + from_depot - direct);
                        }
                    }

                    break_deltas.sort_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(Ordering::Equal)
                    });
                    for delta in break_deltas.into_iter().take(required_breaks) {
                        route_cost += delta;
                    }
                }
            }
        }

        // The route estimate supplies ordering sensitivity, while the MST
        // stabilizes it against nearest-neighbour mistakes and preserves a
        // strong global clustering signal.
        let h = (0.25 * mst + 0.75 * route_cost).max(0.0);
        Ok(if h.is_finite() { h } else { mst.max(0.0) })
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
