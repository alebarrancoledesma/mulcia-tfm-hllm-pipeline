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

use std::cell::RefCell;
use std::collections::HashMap;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const LARGE_PENALTY: f64 = 1.0e12;

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<ExplicitFact>,
    customer_loc_facts: Vec<ExplicitFact>,
    depot_loc_fact: ExplicitFact,

    load_id: usize,
    vehicles_id: usize,
    capacity_id: usize,
    max_vehicles_id: usize,

    demands: Vec<f64>,

    // The final index is the depot. Costs are directed.
    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,
    depot_out_cost: Vec<f64>,

    // A task-dependent directed greedy tour. It is never used blindly:
    // evaluation chooses a state-dependent first customer and filters/rotates
    // this order according to the current unvisited set.
    base_tour: Vec<usize>,
    tour_position: Vec<usize>,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositions: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

fn simple_pne_name(name: &str) -> &str {
    name.trim().strip_prefix("PNE ").unwrap_or(name.trim())
}

fn atom_argument(name: &str, predicate: &str) -> Option<String> {
    let prefix = format!("Atom {}(", predicate);
    name.strip_prefix(&prefix)
        .and_then(|s| s.strip_suffix(')'))
        .map(|s| s.trim().to_string())
}

fn function_argument(name: &str, function: &str) -> Option<String> {
    let simple = simple_pne_name(name);
    let prefix = format!("{}(", function);
    simple.strip_prefix(&prefix)
        .and_then(|s| s.strip_suffix(')'))
        .map(|s| s.trim().to_string())
}

fn binary_function_arguments(name: &str, function: &str) -> Option<(String, String)> {
    let simple = simple_pne_name(name);
    let prefix = format!("{}(", function);
    let inside = simple.strip_prefix(&prefix)?.strip_suffix(')')?;
    let mut parts = inside.split(',').map(str::trim);
    let first = parts.next()?.to_string();
    let second = parts.next()?.to_string();
    if parts.next().is_some() {
        return None;
    }
    Some((first, second))
}

fn operator_tokens(name: &str) -> Vec<&str> {
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
        // --------------------------------------------------------------------
        // Discover positive semantic loc/visited facts. Propositional variable
        // names are intentionally not used because PlanForge names them varN.
        // --------------------------------------------------------------------
        let mut loc_fact_pairs: HashMap<String, (usize, usize)> = HashMap::new();
        let mut visited_fact_pairs: HashMap<String, (usize, usize)> = HashMap::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = atom_argument(&fact_name, "loc") {
                    loc_fact_pairs.insert(object, (var_id, value));
                }
                if let Some(object) = atom_argument(&fact_name, "visited") {
                    visited_fact_pairs.insert(object, (var_id, value));
                }
            }
        }

        if visited_fact_pairs.is_empty() {
            return Err(
                "llm_001 could not discover any positive visited(customer) facts"
                    .to_string()
                    .into(),
            );
        }

        let mut customer_names: Vec<String> = visited_fact_pairs.keys().cloned().collect();
        customer_names.sort();

        let mut customer_index = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            customer_index.insert(name.clone(), index);
        }

        let depot_candidates: Vec<String> = loc_fact_pairs
            .keys()
            .filter(|name| !visited_fact_pairs.contains_key(*name))
            .cloned()
            .collect();

        if depot_candidates.len() != 1 {
            return Err(
                format!(
                    "llm_001 expected one loc-place without a visited fact, found {}",
                    depot_candidates.len()
                )
                .into(),
            );
        }
        let depot_name = depot_candidates[0].clone();

        let depot_pair = loc_fact_pairs.get(&depot_name).ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not recover the depot loc fact"
                    .to_string(),
            )
        })?;
        let depot_loc_fact = ExplicitFact::propositional(depot_pair.0, depot_pair.1);

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        let mut customer_loc_facts = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let visited_pair = visited_fact_pairs.get(name).ok_or_else(|| {
                HeuristicBuildError::from(
                    format!("llm_001 lost visited fact for {}", name),
                )
            })?;
            let loc_pair = loc_fact_pairs.get(name).ok_or_else(|| {
                HeuristicBuildError::from(
                    format!("llm_001 found customer {} without a loc fact", name),
                )
            })?;
            visited_facts.push(ExplicitFact::propositional(
                visited_pair.0,
                visited_pair.1,
            ));
            customer_loc_facts.push(ExplicitFact::propositional(
                loc_pair.0,
                loc_pair.1,
            ));
        }

        // --------------------------------------------------------------------
        // Discover numeric IDs once. Demands and limits are static values, but
        // their task-dependent IDs and values are not hardcoded.
        // --------------------------------------------------------------------
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity_id = None;
        let mut max_vehicles_id = None;
        let mut demand_ids: HashMap<String, usize> = HashMap::new();

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let simple = simple_pne_name(var.name());
            match simple {
                "load()" => load_id = Some(id),
                "vehicles()" => vehicles_id = Some(id),
                "capacity()" => capacity_id = Some(id),
                "max_vehicles()" | "max-vehicles()" => max_vehicles_id = Some(id),
                _ => {
                    if let Some(customer) = function_argument(var.name(), "demand") {
                        demand_ids.insert(customer, id);
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not discover numeric variable load()"
                    .to_string(),
            )
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not discover numeric variable vehicles()"
                    .to_string(),
            )
        })?;
        let capacity_id = capacity_id.ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not discover numeric variable capacity()"
                    .to_string(),
            )
        })?;
        let max_vehicles_id = max_vehicles_id.ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not discover numeric variable max_vehicles()"
                    .to_string(),
            )
        })?;

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut demands = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let id = demand_ids.get(name).ok_or_else(|| {
                HeuristicBuildError::from(
                    format!(
                        "llm_001 could not discover the static demand for customer {}",
                        name
                    ),
                )
            })?;
            let demand = initial_numeric[*id];
            if !demand.is_finite() || demand < 0.0 {
                return Err(
                    format!("llm_001 found invalid demand for customer {}", name).into(),
                );
            }
            demands.push(demand);
        }

        let n = customer_names.len();
        let depot_index = n;
        let mut direct_cost = vec![vec![f64::INFINITY; n + 1]; n + 1];
        let mut via_depot_cost = vec![vec![f64::INFINITY; n]; n];
        let mut return_cost = vec![f64::INFINITY; n];
        let mut depot_out_cost = vec![f64::INFINITY; n];
        direct_cost[depot_index][depot_index] = 0.0;

        // --------------------------------------------------------------------
        // Phase A: primary travel extraction from grounded routing operators.
        // No symmetry is introduced here or later.
        // --------------------------------------------------------------------
        for op in task.get_operators() {
            let tokens = operator_tokens(op.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    let to_index = match customer_index.get(*to) {
                        Some(index) => *index,
                        None => continue,
                    };
                    let from_index = if *from == depot_name {
                        depot_index
                    } else if let Some(index) = customer_index.get(*from) {
                        *index
                    } else {
                        continue;
                    };
                    direct_cost[from_index][to_index] = cost;
                    if from_index == depot_index {
                        depot_out_cost[to_index] = cost;
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(from_index), Some(to_index)) =
                        (customer_index.get(*from), customer_index.get(*to))
                    {
                        via_depot_cost[*from_index][*to_index] = cost;
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(from_index) = customer_index.get(*from) {
                        return_cost[*from_index] = cost;
                        direct_cost[*from_index][depot_index] = cost;
                    }
                }
                _ => {}
            }
        }

        // --------------------------------------------------------------------
        // Phase B: optional PNE fallback for still-missing direct travel arcs.
        // This is deliberately secondary to grounded-operator extraction.
        // --------------------------------------------------------------------
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let Some((from, to)) = binary_function_arguments(var.name(), "travel-cost") else {
                continue;
            };
            let value = initial_numeric[id];
            if !value.is_finite() || value < 0.0 {
                continue;
            }

            let from_index = if from == depot_name {
                Some(depot_index)
            } else {
                customer_index.get(&from).copied()
            };
            let to_index = if to == depot_name {
                Some(depot_index)
            } else {
                customer_index.get(&to).copied()
            };

            if let (Some(i), Some(j)) = (from_index, to_index) {
                if !direct_cost[i][j].is_finite() {
                    direct_cost[i][j] = value;
                }
                if i == depot_index && j < n && !depot_out_cost[j].is_finite() {
                    depot_out_cost[j] = value;
                }
                if i < n && j == depot_index && !return_cost[i].is_finite() {
                    return_cost[i] = value;
                }
            }
        }

        // --------------------------------------------------------------------
        // Phase C: fill only missing arcs via known directed depot legs.
        // --------------------------------------------------------------------
        for from in 0..n {
            for to in 0..n {
                if from == to {
                    direct_cost[from][to] = 0.0;
                    continue;
                }
                let depot_path = return_cost[from] + depot_out_cost[to];
                if !via_depot_cost[from][to].is_finite() && depot_path.is_finite() {
                    via_depot_cost[from][to] = depot_path;
                }
                if !direct_cost[from][to].is_finite()
                    && via_depot_cost[from][to].is_finite()
                {
                    // Conservative last resort: use the legal via-depot cost as
                    // a surrogate for a missing direct arc.
                    direct_cost[from][to] = via_depot_cost[from][to];
                }
            }
        }

        for customer in 0..n {
            if !depot_out_cost[customer].is_finite()
                || !return_cost[customer].is_finite()
            {
                return Err(
                    format!(
                        "llm_001 could not reconstruct directed depot legs for customer {}",
                        customer_names[customer]
                    )
                    .into(),
                );
            }
        }

        let usable_customer_arcs = (0..n).all(|from| {
            (0..n).all(|to| from == to || direct_cost[from][to].is_finite())
        });
        if !usable_customer_arcs {
            return Err(
                "llm_001 could not construct a usable directed routing matrix"
                    .to_string()
                    .into(),
            );
        }

        // Precompute one directed greedy tour for O(n) filtering at evaluation.
        // The state still determines its first customer and rotation.
        let mut base_tour = Vec::with_capacity(n);
        let mut used = vec![false; n];
        let mut current = depot_index;
        for _ in 0..n {
            let mut best = None;
            let mut best_cost = f64::INFINITY;
            for candidate in 0..n {
                if used[candidate] {
                    continue;
                }
                let cost = direct_cost[current][candidate];
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(candidate);
                }
            }
            let next = best.ok_or_else(|| {
                HeuristicBuildError::from(
                    "llm_001 failed while constructing its directed base tour"
                        .to_string(),
                )
            })?;
            used[next] = true;
            base_tour.push(next);
            current = next;
        }

        let mut tour_position = vec![0; n];
        for (position, customer) in base_tour.iter().copied().enumerate() {
            tour_position[customer] = position;
        }

        Ok(Self {
            customer_names,
            visited_facts,
            customer_loc_facts,
            depot_loc_fact,
            load_id,
            vehicles_id,
            capacity_id,
            max_vehicles_id,
            demands,
            direct_cost,
            via_depot_cost,
            return_cost,
            depot_out_cost,
            base_tour,
            tour_position,
            scratch_numeric: RefCell::new(Vec::new()),
            scratch_propositions: RefCell::new(Vec::new()),
            scratch_unvisited: RefCell::new(vec![false; n]),
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

        let _task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let view = registry.view(state);

        // Reuse complete-state scratch storage. Semantic fact objects are used
        // for robust interpretation because SAS variable IDs are task-specific.
        {
            let mut propositions = self.scratch_propositions.borrow_mut();
            state.fill_state(registry, &mut propositions);
        }

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        registry.view(state).fill_numeric(&mut numeric_values);

        let load = numeric_values[self.load_id].max(0.0);
        let vehicles = numeric_values[self.vehicles_id].max(0.0);
        let capacity = numeric_values[self.capacity_id];
        let max_vehicles = numeric_values[self.max_vehicles_id];

        if !load.is_finite()
            || !vehicles.is_finite()
            || !capacity.is_finite()
            || !max_vehicles.is_finite()
            || capacity <= 0.0
            || max_vehicles < 0.0
        {
            return Ok(LARGE_PENALTY);
        }

        let n = self.customer_names.len();
        let depot_index = n;
        let mut unvisited = self.scratch_unvisited.borrow_mut();
        let mut unvisited_count = 0usize;
        let mut remaining_demand = 0.0;

        for customer in 0..n {
            let pending = !self.visited_facts[customer].is_hold(view);
            unvisited[customer] = pending;
            if pending {
                unvisited_count += 1;
                remaining_demand += self.demands[customer];
            }
        }

        let current_location = if self.depot_loc_fact.is_hold(view) {
            Some(depot_index)
        } else {
            self.customer_loc_facts
                .iter()
                .position(|fact| fact.is_hold(view))
        };

        let Some(current_location) = current_location else {
            return Ok(LARGE_PENALTY);
        };

        // Edge case: service is complete but the mandatory final depot return
        // has not yet been taken.
        if unvisited_count == 0 {
            if current_location == depot_index {
                return Ok(0.0);
            }
            let return_cost = self.return_cost[current_location];
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                LARGE_PENALTY
            });
        }

        let residual_capacity = (capacity - load).max(0.0);
        let remaining_vehicle_slots = (max_vehicles - vehicles).max(0.0).floor();
        let total_available_capacity = residual_capacity + remaining_vehicle_slots * capacity;

        // Aggregate fleet-capacity dead-end detection. This depends on the
        // current load and number of vehicles already consumed.
        if remaining_demand > total_available_capacity + 1.0e-9 {
            let excess = remaining_demand - total_available_capacity;
            return Ok(LARGE_PENALTY + excess * 1.0e6);
        }

        // Select a first customer from the actual current state. Direct visits
        // are considered when load permits; otherwise an actual via-depot arc
        // is considered only when another vehicle is available.
        let mut first_customer = None;
        let mut first_cost = f64::INFINITY;
        let mut first_uses_depot = false;

        for customer in 0..n {
            if !unvisited[customer] {
                continue;
            }

            let demand = self.demands[customer];
            let (cost, uses_depot) = if current_location == depot_index {
                if load + demand <= capacity + 1.0e-9 {
                    (self.depot_out_cost[customer], false)
                } else {
                    (f64::INFINITY, false)
                }
            } else if load + demand <= capacity + 1.0e-9 {
                (self.direct_cost[current_location][customer], false)
            } else if remaining_vehicle_slots >= 1.0 {
                (self.via_depot_cost[current_location][customer], true)
            } else {
                (f64::INFINITY, false)
            };

            if cost < first_cost {
                first_cost = cost;
                first_customer = Some(customer);
                first_uses_depot = uses_depot;
            }
        }

        let Some(first_customer) = first_customer else {
            return Ok(LARGE_PENALTY + remaining_demand * 1.0e6);
        };

        // --------------------------------------------------------------------
        // Component 1: state-anchored capacitated routing completion.
        // --------------------------------------------------------------------
        let mut h_routing = first_cost;
        let mut simulated_location = first_customer;
        let mut simulated_load = if first_uses_depot {
            self.demands[first_customer]
        } else {
            load + self.demands[first_customer]
        };
        let mut simulated_vehicles = vehicles + if first_uses_depot { 1.0 } else { 0.0 };
        unvisited[first_customer] = false;

        let anchor = self.tour_position[first_customer];
        for offset in 1..=n {
            let position = (anchor + offset) % n;
            let customer = self.base_tour[position];
            if !unvisited[customer] {
                continue;
            }

            let demand = self.demands[customer];
            if simulated_load + demand <= capacity + 1.0e-9 {
                let cost = self.direct_cost[simulated_location][customer];
                if !cost.is_finite() {
                    return Ok(LARGE_PENALTY);
                }
                h_routing += cost;
                simulated_load += demand;
            } else if simulated_vehicles + 1.0 <= max_vehicles + 1.0e-9 {
                let cost = self.via_depot_cost[simulated_location][customer];
                if !cost.is_finite() {
                    return Ok(LARGE_PENALTY);
                }
                h_routing += cost;
                simulated_load = demand;
                simulated_vehicles += 1.0;
            } else {
                // Aggregate feasibility was checked above. Reaching this branch
                // means the greedy packing order consumed routes poorly. Keep a
                // finite but strong fleet-pressure estimate rather than falsely
                // declaring a proven dead end.
                let surrogate = self.return_cost[simulated_location]
                    + self.depot_out_cost[customer];
                if !surrogate.is_finite() {
                    return Ok(LARGE_PENALTY);
                }
                h_routing += surrogate;
                simulated_load = demand;
                simulated_vehicles += 1.0;
            }

            simulated_location = customer;
            unvisited[customer] = false;
        }

        // Mandatory final return to the single depot.
        let final_return = self.return_cost[simulated_location];
        if !final_return.is_finite() {
            return Ok(LARGE_PENALTY);
        }
        h_routing += final_return;

        // --------------------------------------------------------------------
        // Component 2: capacity/reload pressure.
        // This estimates the unavoidable number of additional routes from the
        // remaining demand after using the current residual load.
        // --------------------------------------------------------------------
        let demand_after_residual = (remaining_demand - residual_capacity).max(0.0);
        let required_new_routes = (demand_after_residual / capacity).ceil();

        let mut cheapest_round_trip = f64::INFINITY;
        for customer in 0..n {
            if self.visited_facts[customer].is_hold(view) {
                continue;
            }
            let round_trip = self.depot_out_cost[customer] + self.return_cost[customer];
            if round_trip < cheapest_round_trip {
                cheapest_round_trip = round_trip;
            }
        }

        let h_capacity = if required_new_routes > 0.0 && cheapest_round_trip.is_finite() {
            // The routing simulation already contains route changes, so use a
            // partial pressure term rather than adding full duplicate tours.
            0.35 * required_new_routes * cheapest_round_trip
        } else {
            0.0
        };

        // --------------------------------------------------------------------
        // Component 3: fleet-slack pressure. States that leave little spare
        // fleet capacity while retaining substantial demand are less promising.
        // --------------------------------------------------------------------
        let fleet_capacity_slack = (total_available_capacity - remaining_demand).max(0.0);
        let normalized_pressure = if total_available_capacity > 0.0 {
            1.0 - fleet_capacity_slack / total_available_capacity
        } else {
            1.0
        };
        let h_vehicles = if cheapest_round_trip.is_finite() {
            0.10 * normalized_pressure * cheapest_round_trip
        } else {
            0.0
        };

        let h = h_routing + h_capacity + h_vehicles;
        Ok(if h.is_finite() && h >= 0.0 {
            h
        } else {
            LARGE_PENALTY
        })
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
