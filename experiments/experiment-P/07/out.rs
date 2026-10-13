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

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let trimmed = name.trim();
    let atom = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);
    if !atom.starts_with(&prefix) || !atom.ends_with(')') {
        return None;
    }

    let argument = &atom[prefix.len()..atom.len() - 1];
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.trim().to_string())
    }
}

fn numeric_core_name(name: &str) -> &str {
    let trimmed = name.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed).trim()
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_matches(|c| c == '(' || c == ')')
        .split_whitespace()
        .collect()
}

fn finite_cost(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<(usize, usize)>,
    location_facts: Vec<(usize, usize)>,
    direct_cost: Vec<Vec<f64>>,
    reset_cost: Vec<Vec<f64>>,
    demand: Vec<f64>,
    capacity: Option<f64>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    max_vehicles: Option<f64>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        for goal_id in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_id);
            let name = task.get_fact_name(&fact).to_string();

            if let Some(customer) = atom_argument(&name, "visited") {
                if !customer_names.iter().any(|x| x == &customer) {
                    customer_names.push(customer);
                }
            } else if let Some(place) = atom_argument(&name, "loc") {
                depot_name = Some(place);
            }
        }

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not identify CVRP visited goals".to_string(),
            ));
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not identify the depot location goal"
                .to_string()
                .into()
        })?;

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());

        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let mut visited_by_name: HashMap<String, (usize, usize)> = HashMap::new();
        let mut location_by_name: HashMap<String, (usize, usize)> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(customer) = atom_argument(&fact_name, "visited") {
                    visited_by_name.insert(customer, (var_id, value));
                }
                if let Some(place) = atom_argument(&fact_name, "loc") {
                    location_by_name.insert(place, (var_id, value));
                }
            }
        }

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        for customer in &customer_names {
            let fact = visited_by_name.get(customer).copied().ok_or_else(
                || -> HeuristicBuildError {
                    format!("llm_001 is missing visited fact for {customer}").into()
                },
            )?;
            visited_facts.push(fact);
        }

        let mut location_facts = Vec::with_capacity(node_names.len());
        for place in &node_names {
            let fact = location_by_name.get(place).copied().ok_or_else(
                || -> HeuristicBuildError {
                    format!("llm_001 is missing location fact for {place}").into()
                },
            )?;
            location_facts.push(fact);
        }

        let node_count = node_names.len();
        let mut direct_cost = vec![vec![f64::INFINITY; node_count]; node_count];
        let mut reset_cost = vec![vec![f64::INFINITY; node_count]; node_count];
        for i in 0..node_count {
            direct_cost[i][i] = 0.0;
            reset_cost[i][i] = 0.0;
        }

        for op in task.get_operators() {
            let tokens = operator_tokens(op.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = match finite_cost(metric_operator_cost_from_initial_values(task, op)) {
                Some(value) => value,
                None => continue,
            };

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        direct_cost[from][to] = direct_cost[from][to].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        reset_cost[from][to] = reset_cost[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = node_index.get(tokens[1]) {
                        direct_cost[from][0] = direct_cost[from][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Reconstruct a missing via-depot cost only from known grounded routing
        // costs. This avoids inspecting fragile derived numeric expressions.
        for from in 1..node_count {
            for to in 1..node_count {
                if from != to && !reset_cost[from][to].is_finite() {
                    let left = direct_cost[from][0];
                    let right = direct_cost[0][to];
                    if left.is_finite() && right.is_finite() {
                        reset_cost[from][to] = left + right;
                    }
                }
            }
        }

        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity_id = None;
        let mut max_vehicles_id = None;
        let mut demand_ids: HashMap<String, usize> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = numeric_core_name(variable.name());
            if name == "load()" {
                load_id = Some(id);
            } else if name == "vehicles()" {
                vehicles_id = Some(id);
            } else if name == "capacity()" {
                capacity_id = Some(id);
            } else if name == "max_vehicles()" {
                max_vehicles_id = Some(id);
            } else {
                for customer in &customer_names {
                    if name == format!("demand({customer})") {
                        demand_ids.insert(customer.clone(), id);
                        break;
                    }
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0);
        let max_vehicles = max_vehicles_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0);

        // A zero demand is the required permissive fallback when an exact
        // demand PNE was folded away by the translator.
        let mut demand = vec![0.0_f64; customer_names.len()];
        for (customer_id, customer) in customer_names.iter().enumerate() {
            if let Some(&numeric_id) = demand_ids.get(customer) {
                if let Some(value) = initial_numeric.get(numeric_id) {
                    if value.is_finite() && *value >= 0.0 {
                        demand[customer_id] = *value;
                    }
                }
            }
        }

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts,
            direct_cost,
            reset_cost,
            demand,
            capacity,
            load_id,
            vehicles_id,
            max_vehicles,
        })
    }

    fn edge_cost(&self, from: usize, to: usize) -> f64 {
        let forward = self.direct_cost[from][to];
        let backward = self.direct_cost[to][from];
        if forward.is_finite() && backward.is_finite() {
            forward.min(backward)
        } else if forward.is_finite() {
            forward
        } else {
            backward
        }
    }

    fn mst_bound(&self, current: usize, unvisited: &[usize]) -> f64 {
        let mut nodes = Vec::with_capacity(unvisited.len() + 2);
        nodes.push(0usize);
        if current != 0 {
            nodes.push(current);
        }
        for &customer in unvisited {
            let node = customer + 1;
            if !nodes.iter().any(|&existing| existing == node) {
                nodes.push(node);
            }
        }

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
            let mut selected_cost = f64::INFINITY;
            for i in 0..count {
                if !used[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let selected = match selected {
                Some(value) => value,
                None => return total,
            };

            used[selected] = true;
            total += selected_cost;

            for next in 0..count {
                if !used[next] {
                    let edge = self.edge_cost(nodes[selected], nodes[next]);
                    if edge < best[next] {
                        best[next] = edge;
                    }
                }
            }
        }

        total
    }

    fn greedy_completion(
        &self,
        current: usize,
        unvisited: &[usize],
        initial_load: f64,
        current_vehicles: Option<f64>,
    ) -> Option<f64> {
        let mut remaining = unvisited.to_vec();
        let mut position = current;
        let mut load = initial_load.max(0.0);
        let mut vehicles = current_vehicles.unwrap_or(0.0);
        let mut total: f64 = 0.0;

        while !remaining.is_empty() {
            let mut best_position: Option<usize> = None;
            let mut best_cost = f64::INFINITY;
            let mut best_uses_reset = false;

            for (remaining_position, &customer_id) in remaining.iter().enumerate() {
                let to = customer_id + 1;
                let customer_demand = self.demand[customer_id];

                let direct_feasible = match self.capacity {
                    Some(capacity) if customer_demand > 0.0 => {
                        load + customer_demand <= capacity + 1.0e-9
                    }
                    _ => true,
                };

                if direct_feasible {
                    let cost = self.direct_cost[position][to];
                    if cost < best_cost {
                        best_cost = cost;
                        best_position = Some(remaining_position);
                        best_uses_reset = false;
                    }
                }

                if position != 0 {
                    let reset_allowed = match self.max_vehicles {
                        Some(maximum) if current_vehicles.is_some() => {
                            vehicles + 1.0 <= maximum + 1.0e-9
                        }
                        _ => true,
                    };
                    let cost = self.reset_cost[position][to];
                    if reset_allowed && cost < best_cost {
                        best_cost = cost;
                        best_position = Some(remaining_position);
                        best_uses_reset = true;
                    }
                }
            }

            let chosen_position = best_position?;
            if !best_cost.is_finite() {
                return None;
            }

            let customer_id = remaining.swap_remove(chosen_position);
            total += best_cost;
            position = customer_id + 1;

            if best_uses_reset {
                vehicles += 1.0;
                load = self.demand[customer_id];
            } else {
                load += self.demand[customer_id];
            }
        }

        let return_cost = self.direct_cost[position][0];
        if !return_cost.is_finite() {
            None
        } else {
            Some(total + return_cost)
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut propositional_values = vec![0usize; task.variables().len()];
        state.fill_state(registry, &mut propositional_values);

        let mut unvisited = Vec::with_capacity(self.customer_names.len());
        for (customer_id, &(var_id, value)) in self.visited_facts.iter().enumerate() {
            if propositional_values[var_id] != value {
                unvisited.push(customer_id);
            }
        }

        let mut current: Option<usize> = None;
        for (node, &(var_id, value)) in self.location_facts.iter().enumerate() {
            if propositional_values[var_id] == value {
                current = Some(node);
                break;
            }
        }
        let current = current.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "llm_001 could not determine the current CVRP location".to_string(),
            )
        })?;

        if unvisited.is_empty() {
            let return_cost = self.direct_cost[current][0];
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                0.0
            });
        }

        let mut numeric_values = vec![0.0_f64; task.numeric_variables().len()];
        registry
            .view(state)
            .fill_numeric(&mut numeric_values)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

        let load = self
            .load_id
            .and_then(|id| numeric_values.get(id).copied())
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        let vehicles = self
            .vehicles_id
            .and_then(|id| numeric_values.get(id).copied())
            .filter(|value| value.is_finite());

        let mst = self.mst_bound(current, &unvisited);

        // Capacity-weighted radial estimate. It is used only when the exact
        // capacity PNE survived translation and at least one exact demand is
        // available. The ordinary routing estimate remains independent of it.
        let mut radial: f64 = 0.0;
        if let Some(capacity) = self.capacity {
            for &customer_id in &unvisited {
                let depot_distance = self.edge_cost(0, customer_id + 1);
                if depot_distance.is_finite() {
                    radial += 2.0 * self.demand[customer_id] * depot_distance / capacity;
                }
            }
        }

        let lower_structure = mst.max(radial).max(0.0);
        let greedy = self.greedy_completion(current, &unvisited, load, vehicles);

        // The MST/radial component supplies a stable global routing signal,
        // while the capacity-aware greedy completion provides a sharper,
        // deliberately non-admissible estimate and distinguishes route orderings.
        let h = match greedy {
            Some(value) if value.is_finite() => {
                lower_structure.max(0.35 * lower_structure + 0.65 * value)
            }
            _ => lower_structure,
        };

        Ok(if h.is_finite() { h.max(0.0) } else { 0.0 })
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
