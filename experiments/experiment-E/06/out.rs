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

const INF: f64 = f64::INFINITY;

#[derive(Clone)]
struct ResourceInfo {
    capacity: f64,
    demand_by_node: Vec<f64>,
    load_id: usize,
    vehicles_id: Option<usize>,
    max_vehicles: Option<f64>,
}

struct LlmHeuristic {
    node_names: Vec<String>,
    depot: usize,
    visited_facts: Vec<(usize, usize)>,
    loc_facts: Vec<(usize, usize)>,
    direct_cost: Vec<f64>,
    prop_count: usize,
    numeric_count: usize,
    resources: Option<ResourceInfo>,
}

fn atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let trimmed = name.trim();
    let rest = trimmed.strip_prefix("Atom ")?;
    let prefix = format!("{}(", predicate);
    let argument = rest.strip_prefix(&prefix)?.strip_suffix(')')?;
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.trim())
    }
}

fn is_exact_numeric_name(actual: &str, expected: &str) -> bool {
    let trimmed = actual.trim();
    trimmed == expected
        || trimmed
            .strip_prefix("PNE ")
            .map(|rest| rest.trim() == expected)
            .unwrap_or(false)
}

fn parse_operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

impl LlmHeuristic {
    fn edge_index(&self, from: usize, to: usize) -> usize {
        from * self.node_names.len() + to
    }

    fn edge(&self, from: usize, to: usize) -> f64 {
        if from == to {
            0.0
        } else {
            self.direct_cost[self.edge_index(from, to)]
        }
    }

    fn build_insertion_path(&self, current: usize, unvisited: &[usize]) -> Vec<usize> {
        let mut path = Vec::with_capacity(unvisited.len() + 2);
        path.push(current);
        path.push(self.depot);

        let mut remaining = unvisited.to_vec();
        while !remaining.is_empty() {
            let mut best_remaining = 0usize;
            let mut best_position = 1usize;
            let mut best_delta = INF;

            for (remaining_pos, &customer) in remaining.iter().enumerate() {
                for position in 1..path.len() {
                    let from = path[position - 1];
                    let to = path[position];
                    let first = self.edge(from, customer);
                    let second = self.edge(customer, to);
                    let replaced = self.edge(from, to);

                    if first.is_finite() && second.is_finite() {
                        let delta = if replaced.is_finite() {
                            first + second - replaced
                        } else {
                            first + second
                        };
                        if delta < best_delta {
                            best_delta = delta;
                            best_remaining = remaining_pos;
                            best_position = position;
                        }
                    }
                }
            }

            if !best_delta.is_finite() {
                best_remaining = 0;
                best_position = path.len() - 1;
            }

            let customer = remaining.swap_remove(best_remaining);
            path.insert(best_position, customer);
        }

        path
    }

    fn path_cost(&self, path: &[usize]) -> f64 {
        let mut total: f64 = 0.0;
        for pair in path.windows(2) {
            let cost = self.edge(pair[0], pair[1]);
            if !cost.is_finite() {
                return INF;
            }
            total += cost;
        }
        total
    }

    fn segment_cost(&self, start: usize, order: &[usize], begin: usize, end: usize) -> f64 {
        if begin == end {
            return self.edge(start, self.depot);
        }

        let mut total = self.edge(start, order[begin]);
        if !total.is_finite() {
            return INF;
        }

        for index in (begin + 1)..end {
            let edge = self.edge(order[index - 1], order[index]);
            if !edge.is_finite() {
                return INF;
            }
            total += edge;
        }

        let return_cost = self.edge(order[end - 1], self.depot);
        if !return_cost.is_finite() {
            INF
        } else {
            total + return_cost
        }
    }

    fn resource_aware_cost(
        &self,
        current: usize,
        order: &[usize],
        resources: &ResourceInfo,
        load: f64,
        vehicles: Option<f64>,
    ) -> f64 {
        let count = order.len();
        let width = count + 1;
        let mut dp = vec![INF; width * width];

        let initial_load = load.max(0.0);
        let first_empty_cost = self.edge(current, self.depot);
        if first_empty_cost.is_finite() {
            dp[0] = first_empty_cost;
        }

        let mut first_demand: f64 = 0.0;
        for end in 1..=count {
            first_demand += resources.demand_by_node[order[end - 1]].max(0.0);
            if initial_load + first_demand > resources.capacity + 1.0e-9 {
                break;
            }
            let cost = self.segment_cost(current, order, 0, end);
            if cost.is_finite() {
                dp[end * width] = cost;
            }
        }

        for begin in 0..count {
            for additional_routes in 0..count {
                let base = dp[begin * width + additional_routes];
                if !base.is_finite() {
                    continue;
                }

                let mut demand: f64 = 0.0;
                for end in (begin + 1)..=count {
                    demand += resources.demand_by_node[order[end - 1]].max(0.0);
                    if demand > resources.capacity + 1.0e-9 {
                        break;
                    }

                    let route_cost = self.segment_cost(self.depot, order, begin, end);
                    if route_cost.is_finite() {
                        let target = end * width + additional_routes + 1;
                        dp[target] = dp[target].min(base + route_cost);
                    }
                }
            }
        }

        let mut unrestricted: f64 = INF;
        let mut restricted: f64 = INF;
        for additional_routes in 0..=count {
            let value = dp[count * width + additional_routes];
            unrestricted = unrestricted.min(value);

            let allowed = match (vehicles, resources.max_vehicles) {
                (Some(used), Some(maximum)) => {
                    used + additional_routes as f64 <= maximum + 1.0e-9
                }
                _ => true,
            };
            if allowed {
                restricted = restricted.min(value);
            }
        }

        if restricted.is_finite() {
            restricted
        } else if unrestricted.is_finite() {
            // The fixed geometric order may require too many routes even when a
            // different packing is feasible. Penalize it rather than claiming
            // that the state is a dead end.
            unrestricted * 1.2
        } else {
            INF
        }
    }

    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let prop_count = task.variables().len();
        let numeric_count = task.numeric_variables().len();

        let mut customer_names: Vec<String> = Vec::new();
        let mut all_loc_names: Vec<String> = Vec::new();

        for var_id in 0..prop_count {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(customer) = atom_argument(fact_name, "visited") {
                    customer_names.push(customer.to_string());
                }
                if let Some(place) = atom_argument(fact_name, "loc") {
                    all_loc_names.push(place.to_string());
                }
            }
        }

        customer_names.sort();
        customer_names.dedup();
        all_loc_names.sort();
        all_loc_names.dedup();

        let mut goal_depot: Option<String> = None;
        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            let goal_name = task.get_fact_name(&goal);
            if let Some(place) = atom_argument(goal_name, "loc") {
                if !customer_names.iter().any(|customer| customer == place) {
                    goal_depot = Some(place.to_string());
                    break;
                }
            }
        }

        let depot_name = goal_depot.or_else(|| {
            all_loc_names
                .iter()
                .find(|place| !customer_names.iter().any(|customer| customer == *place))
                .cloned()
        }).ok_or_else(|| -> HeuristicBuildError {
            "missing depot location fact in CVRP task".to_string().into()
        })?;

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names.iter().cloned());
        let depot = 0usize;

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        for (node, name) in node_names.iter().enumerate() {
            node_by_name.insert(name.clone(), node);
        }

        let mut visited_facts = vec![(usize::MAX, usize::MAX); node_names.len()];
        let mut loc_facts = vec![(usize::MAX, usize::MAX); node_names.len()];

        for var_id in 0..prop_count {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = atom_argument(fact_name, "visited") {
                    if let Some(&node) = node_by_name.get(customer) {
                        visited_facts[node] = (var_id, value);
                    }
                }
                if let Some(place) = atom_argument(fact_name, "loc") {
                    if let Some(&node) = node_by_name.get(place) {
                        loc_facts[node] = (var_id, value);
                    }
                }
            }
        }

        for node in 0..node_names.len() {
            if loc_facts[node].0 == usize::MAX {
                return Err(HeuristicBuildError::from(format!(
                    "missing loc fact for CVRP place {}",
                    node_names[node]
                )));
            }
            if node != depot && visited_facts[node].0 == usize::MAX {
                return Err(HeuristicBuildError::from(format!(
                    "missing visited fact for CVRP customer {}",
                    node_names[node]
                )));
            }
        }

        let node_count = node_names.len();
        let mut direct_cost = vec![INF; node_count * node_count];
        for node in 0..node_count {
            direct_cost[node * node_count + node] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens = parse_operator_tokens(operator.name());
            let metric_cost = metric_operator_cost_from_initial_values(task, operator);
            if !metric_cost.is_finite() || metric_cost < 0.0 {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_node), Some(&to_node)) =
                        (node_by_name.get(*from), node_by_name.get(*to))
                    {
                        let index = from_node * node_count + to_node;
                        direct_cost[index] = direct_cost[index].min(metric_cost);
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_node) = node_by_name.get(*from) {
                        let index = from_node * node_count + depot;
                        direct_cost[index] = direct_cost[index].min(metric_cost);
                    }
                }
                _ => {}
            }
        }

        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut max_vehicles_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> = vec![None; node_count];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if is_exact_numeric_name(name, "load()") {
                load_id = Some(id);
            } else if is_exact_numeric_name(name, "vehicles()") {
                vehicles_id = Some(id);
            } else if is_exact_numeric_name(name, "capacity()") {
                capacity_id = Some(id);
            } else if is_exact_numeric_name(name, "max_vehicles()") {
                max_vehicles_id = Some(id);
            } else {
                for node in 1..node_count {
                    let expected = format!("demand({})", node_names[node]);
                    if is_exact_numeric_name(name, &expected) {
                        demand_ids[node] = Some(id);
                        break;
                    }
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let resources = match (load_id, capacity_id) {
            (Some(load), Some(capacity_var))
                if capacity_var < initial_numeric.len()
                    && demand_ids.iter().skip(1).all(|id| id.is_some()) =>
            {
                let capacity = initial_numeric[capacity_var];
                if capacity.is_finite() && capacity > 0.0 {
                    let mut demand_by_node = vec![0.0; node_count];
                    let mut valid = true;
                    for node in 1..node_count {
                        let id = demand_ids[node].unwrap();
                        if id >= initial_numeric.len()
                            || !initial_numeric[id].is_finite()
                            || initial_numeric[id] < 0.0
                        {
                            valid = false;
                            break;
                        }
                        demand_by_node[node] = initial_numeric[id];
                    }

                    if valid {
                        let max_vehicles = max_vehicles_id.and_then(|id| {
                            initial_numeric.get(id).copied().filter(|value| {
                                value.is_finite() && *value >= 0.0
                            })
                        });
                        Some(ResourceInfo {
                            capacity,
                            demand_by_node,
                            load_id: load,
                            vehicles_id,
                            max_vehicles,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        Ok(Self {
            node_names,
            depot,
            visited_facts,
            loc_facts,
            direct_cost,
            prop_count,
            numeric_count,
            resources,
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

        let mut propositional_values = vec![0usize; self.prop_count];
        state.fill_state(registry, &mut propositional_values);

        let mut current = self.depot;
        for node in 0..self.node_names.len() {
            let (var_id, value) = self.loc_facts[node];
            if propositional_values[var_id] == value {
                current = node;
                break;
            }
        }

        let mut unvisited = Vec::with_capacity(self.node_names.len().saturating_sub(1));
        for node in 1..self.node_names.len() {
            let (var_id, value) = self.visited_facts[node];
            if propositional_values[var_id] != value {
                unvisited.push(node);
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

        let path = self.build_insertion_path(current, &unvisited);
        let geometric_cost = self.path_cost(&path);

        let mut estimate = geometric_cost;
        if let Some(resources) = &self.resources {
            let mut numeric_values = vec![0.0; self.numeric_count];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            let load = numeric_values
                .get(resources.load_id)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.0);
            let vehicles = resources.vehicles_id.and_then(|id| {
                numeric_values
                    .get(id)
                    .copied()
                    .filter(|value| value.is_finite())
            });

            let order = &path[1..path.len() - 1];
            let constrained = self.resource_aware_cost(
                current,
                order,
                resources,
                load,
                vehicles,
            );
            if constrained.is_finite() {
                estimate = constrained;
            } else if geometric_cost.is_finite() {
                estimate = geometric_cost * 1.5;
            }
        }

        if !estimate.is_finite() {
            estimate = 0.0;
        }

        Ok(estimate.max(0.0))
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
