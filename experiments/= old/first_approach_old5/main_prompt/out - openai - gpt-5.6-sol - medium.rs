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

const INF: f64 = f64::INFINITY;

fn unary_atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let name = name.trim();
    let with_atom = format!("Atom {}(", predicate);
    let without_atom = format!("{}(", predicate);

    let rest = if let Some(rest) = name.strip_prefix(&with_atom) {
        rest
    } else if let Some(rest) = name.strip_prefix(&without_atom) {
        rest
    } else {
        return None;
    };

    rest.strip_suffix(')')
}

fn normalized_operator_tokens(name: &str) -> Vec<&str> {
    name.split_whitespace()
        .map(|token| token.trim_matches(|c| c == '(' || c == ')'))
        .filter(|token| !token.is_empty())
        .collect()
}

fn exact_numeric_id(task: &dyn AbstractNumericTask, alternatives: &[String]) -> Option<usize> {
    task.numeric_variables()
        .iter()
        .enumerate()
        .find_map(|(id, variable)| {
            let name = variable.name().trim();
            if alternatives.iter().any(|candidate| name == candidate) {
                Some(id)
            } else {
                None
            }
        })
}

struct LlmHeuristic {
    customer_count: usize,
    visited_facts: Vec<(usize, usize)>,
    customer_loc_facts: Vec<(usize, usize)>,
    depot_loc_fact: (usize, usize),

    // Row-major customer-to-customer costs.
    direct_cost: Vec<f64>,
    via_depot_cost: Vec<f64>,

    // Depot-to-customer and customer-to-depot costs.
    depot_out_cost: Vec<f64>,
    depot_return_cost: Vec<f64>,

    // Optional resource information. These are only enabled when every
    // required exact PNE name survived translation.
    load_id: Option<usize>,
    capacity: Option<f64>,
    demands: Vec<f64>,
    resource_complete: bool,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names = Vec::<String>::new();
        let mut customer_visited = Vec::<(usize, usize)>::new();

        // Customer identities and the truth values of visited predicates are
        // recovered from semantic fact names, never from variable IDs or
        // assumed Boolean encodings.
        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(argument) = unary_atom_argument(fact_name, "visited") {
                    customer_names.push(argument.to_string());
                    customer_visited.push((var_id, value));
                }
            }
        }

        if customer_names.is_empty() {
            return Err("`llm_001` could not identify any CVRP customers"
                .to_string()
                .into());
        }

        let mut customer_index = HashMap::<String, usize>::new();
        for (index, name) in customer_names.iter().enumerate() {
            customer_index.insert(name.clone(), index);
        }

        let n = customer_names.len();
        let unset_fact = (usize::MAX, usize::MAX);
        let mut customer_loc_facts = vec![unset_fact; n];
        let mut depot_name: Option<String> = None;
        let mut depot_loc_fact: Option<(usize, usize)> = None;

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(argument) = unary_atom_argument(fact_name, "loc") {
                    if let Some(&customer) = customer_index.get(argument) {
                        customer_loc_facts[customer] = (var_id, value);
                    } else {
                        // The domain has one depot and every other place is a
                        // customer. Thus the non-customer location is d1,
                        // without hardcoding that object name.
                        depot_name = Some(argument.to_string());
                        depot_loc_fact = Some((var_id, value));
                    }
                }
            }
        }

        if customer_loc_facts.iter().any(|fact| *fact == unset_fact) {
            return Err("`llm_001` could not identify all customer locations"
                .to_string()
                .into());
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "`llm_001` could not identify the depot location"
                .to_string()
                .into()
        })?;
        let depot_loc_fact = depot_loc_fact.ok_or_else(|| -> HeuristicBuildError {
            "`llm_001` could not identify the depot location fact"
                .to_string()
                .into()
        })?;

        let mut direct_cost = vec![INF; n * n];
        let mut via_depot_cost = vec![INF; n * n];
        let mut depot_out_cost = vec![INF; n];
        let mut depot_return_cost = vec![INF; n];

        // Routing costs are reconstructed from grounded operators. This is
        // robust to travel-cost PNEs being folded into derived expressions.
        for operator in task.get_operators() {
            let tokens = normalized_operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    let from = tokens[1];
                    let to = tokens[2];
                    let Some(&to_index) = customer_index.get(to) else {
                        continue;
                    };

                    if from == depot_name {
                        depot_out_cost[to_index] = depot_out_cost[to_index].min(cost);
                    } else if let Some(&from_index) = customer_index.get(from) {
                        let offset = from_index * n + to_index;
                        direct_cost[offset] = direct_cost[offset].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    let Some(&from_index) = customer_index.get(tokens[1]) else {
                        continue;
                    };
                    let Some(&to_index) = customer_index.get(tokens[2]) else {
                        continue;
                    };
                    let offset = from_index * n + to_index;
                    via_depot_cost[offset] = via_depot_cost[offset].min(cost);
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from_index) = customer_index.get(tokens[1]) {
                        depot_return_cost[from_index] =
                            depot_return_cost[from_index].min(cost);
                    }
                }
                _ => {}
            }
        }

        let load_id = exact_numeric_id(
            task,
            &["load()".to_string(), "PNE load()".to_string()],
        );
        let capacity_id = exact_numeric_id(
            task,
            &["capacity()".to_string(), "PNE capacity()".to_string()],
        );

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0);

        let mut demands = vec![0.0; n];
        let mut all_demands_found = true;
        for (customer, name) in customer_names.iter().enumerate() {
            let plain = format!("demand({})", name);
            let prefixed = format!("PNE demand({})", name);
            let demand_id = exact_numeric_id(task, &[plain, prefixed]);

            if let Some(value) = demand_id
                .and_then(|id| initial_numeric.get(id).copied())
                .filter(|value| value.is_finite() && *value >= 0.0)
            {
                demands[customer] = value;
            } else {
                // Demand is deliberately treated as an optional signal: the
                // translator may legally fold it into a derived comparison.
                all_demands_found = false;
            }
        }

        let resource_complete =
            load_id.is_some() && capacity.is_some() && all_demands_found;

        Ok(Self {
            customer_count: n,
            visited_facts: customer_visited,
            customer_loc_facts,
            depot_loc_fact,
            direct_cost,
            via_depot_cost,
            depot_out_cost,
            depot_return_cost,
            load_id,
            capacity,
            demands,
            resource_complete,
        })
    }

    fn direct(&self, from: usize, to: usize) -> f64 {
        self.direct_cost[from * self.customer_count + to]
    }

    fn via_depot(&self, from: usize, to: usize) -> f64 {
        self.via_depot_cost[from * self.customer_count + to]
    }

    fn relaxed_edge(&self, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }

        let depot = self.customer_count;
        if a == depot {
            return self.depot_out_cost[b].min(self.depot_return_cost[b]);
        }
        if b == depot {
            return self.depot_out_cost[a].min(self.depot_return_cost[a]);
        }

        self.direct(a, b)
            .min(self.direct(b, a))
            .min(self.via_depot(a, b))
            .min(self.via_depot(b, a))
    }

    fn mst_estimate(&self, active: &[usize]) -> f64 {
        if active.len() <= 1 {
            return 0.0;
        }

        let count = active.len();
        let mut used = vec![false; count];
        let mut key = vec![INF; count];
        key[0] = 0.0;
        let mut total = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_key = INF;
            for i in 0..count {
                if !used[i] && key[i] < selected_key {
                    selected = Some(i);
                    selected_key = key[i];
                }
            }

            let Some(u) = selected else {
                break;
            };
            used[u] = true;
            total += selected_key;

            for v in 0..count {
                if !used[v] {
                    let edge = self.relaxed_edge(active[u], active[v]);
                    if edge < key[v] {
                        key[v] = edge;
                    }
                }
            }
        }

        if total.is_finite() { total } else { 0.0 }
    }

    fn greedy_completion(
        &self,
        current_location: usize,
        unvisited: &[bool],
        initial_load: f64,
    ) -> f64 {
        let n = self.customer_count;
        let depot = n;
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|value| **value).count();
        let mut current = current_location;
        let mut load = initial_load;
        let mut total = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = INF;
            let mut best_uses_depot = false;

            for customer in 0..n {
                if !remaining[customer] {
                    continue;
                }

                let (candidate_cost, uses_depot) = if current == depot {
                    (self.depot_out_cost[customer], false)
                } else if self.resource_complete {
                    let capacity = self.capacity.unwrap_or(f64::MAX);
                    let direct_feasible =
                        load + self.demands[customer] <= capacity + 1.0e-9;
                    let direct = if direct_feasible {
                        self.direct(current, customer)
                    } else {
                        INF
                    };
                    let via = self.via_depot(current, customer);
                    if via < direct {
                        (via, true)
                    } else {
                        (direct, false)
                    }
                } else {
                    let direct = self.direct(current, customer);
                    let via = self.via_depot(current, customer);
                    if via < direct {
                        (via, true)
                    } else {
                        (direct, false)
                    }
                };

                if candidate_cost < best_cost {
                    best_cost = candidate_cost;
                    best_customer = Some(customer);
                    best_uses_depot = uses_depot;
                }
            }

            let Some(next) = best_customer else {
                // In a malformed or partially reconstructed cost graph, keep
                // the heuristic finite rather than poisoning the search.
                break;
            };

            total += best_cost;
            remaining[next] = false;
            remaining_count -= 1;

            if self.resource_complete {
                if current == depot || best_uses_depot {
                    load = self.demands[next];
                } else {
                    load += self.demands[next];
                }
            }
            current = next;
        }

        if current != depot {
            let return_cost = self.depot_return_cost[current];
            if return_cost.is_finite() {
                total += return_cost;
            }
        }

        if total.is_finite() { total } else { 0.0 }
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

        let n = self.customer_count;
        let depot = n;
        let mut unvisited = vec![false; n];
        let mut unvisited_count = 0usize;

        for customer in 0..n {
            let (var_id, true_value) = self.visited_facts[customer];
            let visited = propositional_values[var_id] == true_value;
            unvisited[customer] = !visited;
            if !visited {
                unvisited_count += 1;
            }
        }

        let mut current_location = depot;
        let (depot_var, depot_value) = self.depot_loc_fact;
        if propositional_values[depot_var] != depot_value {
            for customer in 0..n {
                let (var_id, location_value) = self.customer_loc_facts[customer];
                if propositional_values[var_id] == location_value {
                    current_location = customer;
                    break;
                }
            }
        }

        // If all customers have been served, the only remaining action is the
        // exact return-to-depot action.
        if unvisited_count == 0 {
            if current_location == depot {
                return Ok(0.0);
            }
            let return_cost = self.depot_return_cost[current_location];
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                0.0
            });
        }

        let mut load = 0.0;
        if self.resource_complete {
            let mut numeric_values = vec![0.0; task.numeric_variables().len()];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
            if let Some(load_id) = self.load_id {
                if let Some(value) = numeric_values.get(load_id) {
                    if value.is_finite() {
                        load = (*value).max(0.0);
                    }
                }
            }
        }

        // Active nodes for the routing relaxation: current endpoint, every
        // unvisited customer, and the depot. Duplicate nodes are avoided.
        let mut active = Vec::with_capacity(unvisited_count + 2);
        active.push(current_location);
        for customer in 0..n {
            if unvisited[customer] && customer != current_location {
                active.push(customer);
            }
        }
        if !active.contains(&depot) {
            active.push(depot);
        }

        let mst = self.mst_estimate(&active);
        let greedy = self.greedy_completion(current_location, &unvisited, load);

        // The greedy completion is more predictive, while the MST component
        // reduces nearest-neighbour myopia and makes the estimate smoother.
        let h = (0.75 * greedy + 0.25 * mst).max(0.0);
        Ok(if h.is_finite() { h } else { 0.0 })
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
