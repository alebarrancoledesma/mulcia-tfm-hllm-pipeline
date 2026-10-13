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
struct StoredFact {
    var: usize,
    value: usize,
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<StoredFact>,
    location_facts: Vec<StoredFact>,
    depot_index: usize,
    distances: Vec<f64>,
    demands: Vec<f64>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    numeric_count: usize,
    resource_information_complete: bool,
}

fn parse_atom_argument(name: &str, predicate: &str) -> Option<String> {
    let prefix = format!("Atom {}(", predicate);
    let text = name.trim();
    let rest = text.strip_prefix(&prefix)?;
    let close = rest.find(')')?;

    if !rest[close + 1..].trim().is_empty() {
        return None;
    }

    let argument = rest[..close].trim();
    if argument.is_empty() || argument.contains(char::is_whitespace) {
        return None;
    }

    Some(argument.to_string())
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .map(|token| token.trim_matches(|c| c == '(' || c == ')'))
        .filter(|token| !token.is_empty())
        .collect()
}

fn exact_pne_name(name: &str, target: &str) -> bool {
    let trimmed = name.trim();
    trimmed == target || trimmed == format!("PNE {}", target)
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut visited_by_customer: HashMap<String, StoredFact> = HashMap::new();
        let mut location_by_place: HashMap<String, StoredFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let variable = &task.variables()[var_id];
            for value in 0..variable.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(customer) = parse_atom_argument(&fact_name, "visited") {
                    visited_by_customer.insert(
                        customer,
                        StoredFact {
                            var: var_id,
                            value,
                        },
                    );
                }

                if let Some(place) = parse_atom_argument(&fact_name, "loc") {
                    location_by_place.insert(
                        place,
                        StoredFact {
                            var: var_id,
                            value,
                        },
                    );
                }
            }
        }

        if visited_by_customer.is_empty() {
            return Err("`llm_001` could not identify any CVRP customers"
                .to_string()
                .into());
        }

        let mut customer_names: Vec<String> =
            visited_by_customer.keys().cloned().collect();
        customer_names.sort();

        let depot_name = location_by_place
            .keys()
            .find(|place| !visited_by_customer.contains_key(*place))
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "`llm_001` could not identify the CVRP depot"
                    .to_string()
                    .into()
            })?;

        let customer_count = customer_names.len();
        let place_count = customer_count + 1;
        let depot_index = customer_count;

        let mut place_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            place_index.insert(name.clone(), index);
        }
        place_index.insert(depot_name.clone(), depot_index);

        let mut visited_facts = Vec::with_capacity(customer_count);
        let mut location_facts = Vec::with_capacity(place_count);

        for name in &customer_names {
            let visited = *visited_by_customer.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    "`llm_001` found an inconsistent visited-fact mapping"
                        .to_string()
                        .into()
                },
            )?;
            let location = *location_by_place.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    "`llm_001` found a customer without a location fact"
                        .to_string()
                        .into()
                },
            )?;
            visited_facts.push(visited);
            location_facts.push(location);
        }

        let depot_location = *location_by_place.get(&depot_name).ok_or_else(
            || -> HeuristicBuildError {
                "`llm_001` found no location fact for the depot"
                    .to_string()
                    .into()
            },
        )?;
        location_facts.push(depot_location);

        let mut distances = vec![f64::INFINITY; place_count * place_count];
        for index in 0..place_count {
            distances[index * place_count + index] = 0.0;
        }

        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        let cell = &mut distances[from * place_count + to];
                        if cost < *cell {
                            *cell = cost;
                        }
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = place_index.get(tokens[1]) {
                        let cell = &mut distances[from * place_count + depot_index];
                        if cost < *cell {
                            *cell = cost;
                        }
                    }
                }
                _ => {}
            }
        }

        for customer in 0..customer_count {
            let outward = distances[depot_index * place_count + customer];
            let inward = distances[customer * place_count + depot_index];
            if !outward.is_finite() || !inward.is_finite() {
                return Err(
                    "`llm_001` could not reconstruct all depot routing costs"
                        .to_string()
                        .into(),
                );
            }
        }

        let numeric_count = task.numeric_variables().len();
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut demands = vec![0.0; customer_count];
        let mut demand_found = vec![false; customer_count];

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();

            if exact_pne_name(name, "load()") {
                load_id = Some(id);
            } else if exact_pne_name(name, "vehicles()") {
                vehicles_id = Some(id);
            } else if exact_pne_name(name, "capacity()") {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value > 0.0 {
                        capacity = Some(*value);
                    }
                }
            } else if exact_pne_name(name, "max_vehicles()") {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value >= 0.0 {
                        max_vehicles = Some(*value);
                    }
                }
            } else {
                for (customer, customer_name) in customer_names.iter().enumerate() {
                    let target = format!("demand({})", customer_name);
                    if exact_pne_name(name, &target) {
                        if let Some(value) = initial_numeric.get(id) {
                            if value.is_finite() && *value >= 0.0 {
                                demands[customer] = *value;
                                demand_found[customer] = true;
                            }
                        }
                        break;
                    }
                }
            }
        }

        let resource_information_complete =
            capacity.is_some() && load_id.is_some() && demand_found.iter().all(|x| *x);

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts,
            depot_index,
            distances,
            demands,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
            numeric_count,
            resource_information_complete,
        })
    }

    fn distance(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }

        let place_count = self.customer_names.len() + 1;
        let direct = self.distances[from * place_count + to];
        if direct.is_finite() {
            return direct;
        }

        if from != self.depot_index && to != self.depot_index {
            let to_depot = self.distances[from * place_count + self.depot_index];
            let from_depot = self.distances[self.depot_index * place_count + to];
            if to_depot.is_finite() && from_depot.is_finite() {
                return to_depot + from_depot;
            }
        }

        f64::INFINITY
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
        let customer_count = self.customer_names.len();

        let mut unvisited = vec![false; customer_count];
        let mut remaining = 0usize;

        for customer in 0..customer_count {
            let stored = self.visited_facts[customer];
            let fact = ExplicitFact::propositional(stored.var, stored.value);
            if !fact.is_hold(registry.view(state)) {
                unvisited[customer] = true;
                remaining += 1;
            }
        }

        let mut current = self.depot_index;
        for (place, stored) in self.location_facts.iter().enumerate() {
            let fact = ExplicitFact::propositional(stored.var, stored.value);
            if fact.is_hold(registry.view(state)) {
                current = place;
                break;
            }
        }

        if remaining == 0 {
            let return_cost = self.distance(current, self.depot_index);
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                0.0
            });
        }

        let mut current_load = 0.0;
        let mut current_vehicles = None;

        if self.resource_information_complete {
            let mut numeric_values = vec![0.0; self.numeric_count];
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;

            if let Some(id) = self.load_id {
                current_load = numeric_values[id].max(0.0);
            }
            if let Some(id) = self.vehicles_id {
                current_vehicles = Some(numeric_values[id].max(0.0));
            }

            if let (Some(capacity), Some(max_vehicles), Some(vehicles)) =
                (self.capacity, self.max_vehicles, current_vehicles)
            {
                let total_remaining_demand: f64 = (0..customer_count)
                    .filter(|customer| unvisited[*customer])
                    .map(|customer| self.demands[customer])
                    .sum();

                let room_on_current_vehicle = (capacity - current_load).max(0.0);
                let future_vehicle_count = (max_vehicles - vehicles).max(0.0).floor();
                let available_capacity =
                    room_on_current_vehicle + future_vehicle_count * capacity;

                if total_remaining_demand > available_capacity + 1.0e-9 {
                    return Ok(1.0e12 + total_remaining_demand);
                }
            }
        }

        let capacity = self.capacity.unwrap_or(f64::MAX);
        let use_resources = self.resource_information_complete;
        let mut estimate = 0.0;

        while remaining > 0 {
            let mut best_customer = None;
            let mut best_cost = f64::INFINITY;

            for customer in 0..customer_count {
                if !unvisited[customer] {
                    continue;
                }

                if use_resources
                    && current_load + self.demands[customer] > capacity + 1.0e-9
                {
                    continue;
                }

                let cost = self.distance(current, customer);
                if cost < best_cost {
                    best_cost = cost;
                    best_customer = Some(customer);
                }
            }

            if let Some(customer) = best_customer {
                if !best_cost.is_finite() {
                    break;
                }

                estimate += best_cost;
                current = customer;
                if use_resources {
                    current_load += self.demands[customer];
                }
                unvisited[customer] = false;
                remaining -= 1;
                continue;
            }

            if use_resources && current != self.depot_index {
                let mut restart_customer = None;
                let mut restart_cost = f64::INFINITY;

                for customer in 0..customer_count {
                    if !unvisited[customer] || self.demands[customer] > capacity + 1.0e-9 {
                        continue;
                    }

                    let cost = self.distance(current, self.depot_index)
                        + self.distance(self.depot_index, customer);
                    if cost < restart_cost {
                        restart_cost = cost;
                        restart_customer = Some(customer);
                    }
                }

                if let Some(customer) = restart_customer {
                    if !restart_cost.is_finite() {
                        break;
                    }

                    estimate += restart_cost;
                    current = customer;
                    current_load = self.demands[customer];
                    unvisited[customer] = false;
                    remaining -= 1;
                    continue;
                }
            }

            return Ok(1.0e12 + remaining as f64);
        }

        let return_cost = self.distance(current, self.depot_index);
        if return_cost.is_finite() {
            estimate += return_cost;
        }

        if !estimate.is_finite() || estimate < 0.0 {
            Ok(0.0)
        } else {
            Ok(estimate)
        }
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
