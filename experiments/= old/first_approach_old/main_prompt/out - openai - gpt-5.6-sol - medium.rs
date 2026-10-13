use std::collections::HashMap;

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::task::ExplicitFact;

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
struct CustomerInfo {
    visited_fact: ExplicitFact,
    loc_fact: ExplicitFact,
    demand: f64,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    depot_loc_fact: Option<ExplicitFact>,

    // Node 0 is the depot; customer i is node i + 1.
    distances: Vec<Vec<f64>>,

    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,

    fallback_edge_cost: f64,
}

impl LlmHeuristic {
    fn positive_atom_argument(name: &str, predicate: &str) -> Option<String> {
        let prefix = format!("Atom {}(", predicate);
        if !name.starts_with(&prefix) || !name.ends_with(')') {
            return None;
        }

        let argument = &name[prefix.len()..name.len() - 1];
        if argument.is_empty() {
            None
        } else {
            Some(argument.to_string())
        }
    }

    fn simple_pne_body(name: &str) -> &str {
        name.trim().strip_prefix("PNE ").unwrap_or(name.trim())
    }

    fn parse_zero_arity_pne(name: &str, function: &str) -> bool {
        let body = Self::simple_pne_body(name);
        body == format!("{}()", function)
    }

    fn parse_unary_pne(name: &str, function: &str) -> Option<String> {
        let body = Self::simple_pne_body(name);
        let prefix = format!("{}(", function);
        if !body.starts_with(&prefix) || !body.ends_with(')') {
            return None;
        }

        let argument = body[prefix.len()..body.len() - 1].trim();
        if argument.is_empty() || argument.contains(',') {
            None
        } else {
            Some(argument.to_string())
        }
    }

    fn parse_binary_pne(name: &str, function: &str) -> Option<(String, String)> {
        let body = Self::simple_pne_body(name);
        let prefix = format!("{}(", function);
        if !body.starts_with(&prefix) || !body.ends_with(')') {
            return None;
        }

        let arguments = &body[prefix.len()..body.len() - 1];
        let mut parts = arguments
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|part| !part.is_empty());

        let first = parts.next()?;
        let second = parts.next()?;
        if parts.next().is_some() {
            return None;
        }

        Some((first.to_string(), second.to_string()))
    }

    fn edge_cost(&self, from: usize, to: usize) -> f64 {
        let value = self.distances[from][to];
        if value.is_finite() && value >= 0.0 {
            value
        } else {
            self.fallback_edge_cost
        }
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut visited_facts: HashMap<String, ExplicitFact> = HashMap::new();
        let mut loc_facts: HashMap<String, ExplicitFact> = HashMap::new();

        // Propositional variable names are generic, so inspect semantic fact
        // names for every value in every finite-domain variable.
        for (var_id, variable) in task.variables().iter().enumerate() {
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(fact);

                if let Some(object) = Self::positive_atom_argument(&fact_name, "visited") {
                    visited_facts.insert(object, fact);
                } else if let Some(object) = Self::positive_atom_argument(&fact_name, "loc") {
                    loc_facts.insert(object, fact);
                }
            }
        }

        let mut customer_names: Vec<String> = visited_facts.keys().cloned().collect();
        customer_names.sort();

        // The depot is the unique object with a positive loc fact and no
        // corresponding visited predicate.
        let depot_name = loc_facts
            .keys()
            .find(|name| !visited_facts.contains_key(*name))
            .cloned();
        let depot_loc_fact = depot_name
            .as_ref()
            .and_then(|name| loc_facts.get(name).cloned());

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut demands: HashMap<String, f64> = HashMap::new();
        let mut raw_distances: HashMap<(String, String), f64> = HashMap::new();

        for (numeric_id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            let value = initial_numeric[numeric_id];

            if Self::parse_zero_arity_pne(name, "load") {
                load_id = Some(numeric_id);
            } else if Self::parse_zero_arity_pne(name, "vehicles") {
                vehicles_id = Some(numeric_id);
            } else if Self::parse_zero_arity_pne(name, "capacity") {
                capacity = Some(value);
            } else if Self::parse_zero_arity_pne(name, "max-vehicles")
                || Self::parse_zero_arity_pne(name, "max_vehicles")
            {
                max_vehicles = Some(value);
            } else if let Some(customer) = Self::parse_unary_pne(name, "demand") {
                demands.insert(customer, value);
            } else if let Some((from, to)) = Self::parse_binary_pne(name, "travel-cost") {
                if value.is_finite() && value >= 0.0 {
                    raw_distances.insert((from, to), value);
                }
            }
        }

        let node_count = customer_names.len() + 1;
        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for i in 0..node_count {
            distances[i][i] = 0.0;
        }

        let mut node_names = Vec::with_capacity(node_count);
        node_names.push(depot_name.clone().unwrap_or_else(|| "__depot__".to_string()));
        node_names.extend(customer_names.iter().cloned());

        let mut known_cost_sum = 0.0;
        let mut known_cost_count = 0usize;
        for i in 0..node_count {
            for j in 0..node_count {
                if let Some(cost) = raw_distances.get(&(node_names[i].clone(), node_names[j].clone())) {
                    distances[i][j] = *cost;
                    if i != j && *cost > 0.0 {
                        known_cost_sum += *cost;
                        known_cost_count += 1;
                    }
                }
            }
        }

        let fallback_edge_cost = if known_cost_count > 0 {
            (known_cost_sum / known_cost_count as f64).max(1.0)
        } else {
            1.0
        };

        // Fill missing directed edges conservatively. When both depot legs
        // are known, their sum is exactly the cost of a via-depot transition.
        for i in 0..node_count {
            for j in 0..node_count {
                if !distances[i][j].is_finite() {
                    let via_depot = distances[i][0] + distances[0][j];
                    distances[i][j] = if via_depot.is_finite() {
                        via_depot
                    } else {
                        fallback_edge_cost
                    };
                }
            }
        }

        let mut customers = Vec::with_capacity(customer_names.len());
        for customer_name in customer_names {
            if let (Some(visited_fact), Some(loc_fact)) = (
                visited_facts.get(&customer_name).cloned(),
                loc_facts.get(&customer_name).cloned(),
            ) {
                customers.push(CustomerInfo {
                    visited_fact,
                    loc_fact,
                    demand: demands.get(&customer_name).copied().unwrap_or(0.0),
                });
            }
        }

        Ok(Self {
            customers,
            depot_loc_fact,
            distances,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            fallback_edge_cost,
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
        let state_view = registry.view(state);

        let mut current_node = 0usize;
        if !self
            .depot_loc_fact
            .as_ref()
            .map(|fact| fact.is_hold(state_view))
            .unwrap_or(false)
        {
            for (customer_index, customer) in self.customers.iter().enumerate() {
                if customer.loc_fact.is_hold(state_view) {
                    current_node = customer_index + 1;
                    break;
                }
            }
        }

        let mut remaining = Vec::with_capacity(self.customers.len());
        let mut remaining_demand = 0.0;
        for (customer_index, customer) in self.customers.iter().enumerate() {
            if !customer.visited_fact.is_hold(state_view) {
                remaining.push(customer_index);
                remaining_demand += customer.demand.max(0.0);
            }
        }

        if remaining.is_empty() {
            return Ok(self.edge_cost(current_node, 0));
        }

        let mut numeric_values = Vec::new();
        registry
            .view(state)
            .fill_numeric(&mut numeric_values);

        let mut simulated_load = self
            .load_id
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(0.0)
            .max(0.0);
        let capacity = self.capacity.unwrap_or(f64::INFINITY);
        let vehicles_used = self
            .vehicles_id
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(1.0);

        // Capacity/vehicle resource dead-end test. The current vehicle can
        // carry its residual capacity and every additional vehicle can carry
        // at most one full capacity.
        if capacity.is_finite() && capacity > 0.0 {
            let residual = (capacity - simulated_load).max(0.0);
            let extra_vehicle_capacity = self
                .max_vehicles
                .map(|limit| (limit - vehicles_used).max(0.0) * capacity)
                .unwrap_or(f64::INFINITY);

            if remaining_demand > residual + extra_vehicle_capacity + 1.0e-9 {
                return Ok(1.0e12 + remaining_demand);
            }
        }

        // Greedy capacitated completion. It estimates a concrete continuation
        // from the state's real location and load. Direct visits are preferred
        // while capacity permits; otherwise a via-depot transition is used.
        let mut served = vec![false; self.customers.len()];
        let mut unserved_count = remaining.len();
        let mut active_node = current_node;
        let mut simulated_vehicles = vehicles_used;
        let mut route_cost = 0.0;

        while unserved_count > 0 {
            let mut best_direct: Option<(usize, f64)> = None;

            for &customer_index in &remaining {
                if served[customer_index] {
                    continue;
                }

                let demand = self.customers[customer_index].demand.max(0.0);
                if simulated_load + demand <= capacity + 1.0e-9 {
                    let cost = self.edge_cost(active_node, customer_index + 1);
                    if best_direct
                        .as_ref()
                        .map(|(_, best_cost)| cost < *best_cost)
                        .unwrap_or(true)
                    {
                        best_direct = Some((customer_index, cost));
                    }
                }
            }

            if let Some((customer_index, cost)) = best_direct {
                route_cost += cost;
                simulated_load += self.customers[customer_index].demand.max(0.0);
                active_node = customer_index + 1;
                served[customer_index] = true;
                unserved_count -= 1;
                continue;
            }

            if let Some(limit) = self.max_vehicles {
                if simulated_vehicles + 1.0 > limit + 1.0e-9 {
                    return Ok(1.0e12 + route_cost + unserved_count as f64);
                }
            }

            let mut best_after_depot: Option<(usize, f64)> = None;
            for &customer_index in &remaining {
                if served[customer_index] {
                    continue;
                }

                let cost = self.edge_cost(0, customer_index + 1);
                if best_after_depot
                    .as_ref()
                    .map(|(_, best_cost)| cost < *best_cost)
                    .unwrap_or(true)
                {
                    best_after_depot = Some((customer_index, cost));
                }
            }

            let Some((customer_index, depot_to_customer)) = best_after_depot else {
                break;
            };

            route_cost += self.edge_cost(active_node, 0) + depot_to_customer;
            simulated_vehicles += 1.0;
            simulated_load = self.customers[customer_index].demand.max(0.0);
            active_node = customer_index + 1;
            served[customer_index] = true;
            unserved_count -= 1;
        }

        route_cost += self.edge_cost(active_node, 0);

        let h = if route_cost.is_finite() {
            route_cost.max(0.0)
        } else {
            self.fallback_edge_cost * (remaining.len() as f64 + 1.0)
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
