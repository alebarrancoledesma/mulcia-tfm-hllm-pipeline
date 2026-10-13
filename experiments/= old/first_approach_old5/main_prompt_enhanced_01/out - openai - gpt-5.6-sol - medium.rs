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
use std::cmp::Ordering;
use std::collections::HashMap;


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const LARGE_PENALTY: f64 = 1.0e15;
const NEIGHBOR_PROBES: usize = 6;
const INF: f64 = f64::INFINITY;

#[derive(Clone, Copy)]
struct FactKey {
    var: usize,
    value: usize,
}

impl FactKey {
    fn holds(&self, values: &[usize]) -> bool {
        values.get(self.var).copied() == Some(self.value)
    }
}

fn unary_atom_argument(name: &str, predicate: &str) -> Option<String> {
    let trimmed = name.trim();
    let body = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);
    let argument = body.strip_prefix(&prefix)?.strip_suffix(')')?.trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn numeric_name_is(name: &str, bare: &str) -> bool {
    let trimmed = name.trim();
    trimmed == bare || trimmed == format!("PNE {}", bare)
}

fn parse_travel_cost_name(name: &str) -> Option<(String, String)> {
    let trimmed = name.trim();
    let start = trimmed.find("travel-cost(")?;
    let body = &trimmed[start + "travel-cost(".len()..];
    let arguments = body.strip_suffix(')')?;
    let mut parts = arguments.split(',').map(str::trim);
    let from = parts.next()?;
    let to = parts.next()?;
    if from.is_empty() || to.is_empty() || parts.next().is_some() {
        return None;
    }
    Some((from.to_string(), to.to_string()))
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

struct LlmHeuristic {
    customer_count: usize,
    depot_index: usize,
    visited_facts: Vec<FactKey>,
    location_facts: Vec<FactKey>,

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,

    direct_cost: Vec<f64>,
    via_cost: Vec<f64>,
    nearest_from: Vec<Vec<usize>>,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn matrix_index(&self, from: usize, to: usize) -> usize {
        from * (self.customer_count + 1) + to
    }

    fn direct(&self, from: usize, to: usize) -> f64 {
        self.direct_cost[self.matrix_index(from, to)]
    }

    fn via(&self, from: usize, to: usize) -> f64 {
        self.via_cost[self.matrix_index(from, to)]
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut loc_by_object: HashMap<String, FactKey> = HashMap::new();
        let mut visited_by_object: HashMap<String, FactKey> = HashMap::new();
        let mut customer_names: Vec<String> = Vec::new();

        // Discover loc and visited semantics through fact names, not through
        // generic SAS variable names or assumed truth values.
        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = unary_atom_argument(fact_name, "loc") {
                    loc_by_object.insert(
                        object,
                        FactKey {
                            var: var_id,
                            value,
                        },
                    );
                }

                if let Some(object) = unary_atom_argument(fact_name, "visited") {
                    if !visited_by_object.contains_key(&object) {
                        customer_names.push(object.clone());
                    }
                    visited_by_object.insert(
                        object,
                        FactKey {
                            var: var_id,
                            value,
                        },
                    );
                }
            }
        }

        if customer_names.is_empty() {
            return Err("llm_001 found no visited(customer) facts".to_string().into());
        }

        // The depot is the location object that is not a visitable customer.
        let depot_name = loc_by_object
            .keys()
            .find(|name| !visited_by_object.contains_key(*name))
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not identify the depot from loc facts"
                    .to_string()
                    .into()
            })?;

        let customer_count = customer_names.len();
        let depot_index = customer_count;
        let place_count = customer_count + 1;

        let mut place_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            place_index.insert(name.clone(), index);
        }
        place_index.insert(depot_name.clone(), depot_index);

        let mut visited_facts = Vec::with_capacity(customer_count);
        let mut location_facts = Vec::with_capacity(place_count);
        for name in &customer_names {
            visited_facts.push(*visited_by_object.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    "llm_001 has an inconsistent visited-fact map"
                        .to_string()
                        .into()
                },
            )?);
            location_facts.push(*loc_by_object.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    format!("llm_001 found no loc fact for customer {name}").into()
                },
            )?);
        }
        location_facts.push(*loc_by_object.get(&depot_name).ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 found no loc fact for the depot".to_string().into()
            },
        )?);

        let numeric_variables = task.numeric_variables();
        let initial_numeric = task.get_initial_numeric_state_values();

        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;

        for (id, variable) in numeric_variables.iter().enumerate() {
            let name = variable.name();
            if numeric_name_is(name, "load()") {
                load_id = Some(id);
            } else if numeric_name_is(name, "vehicles()") {
                vehicles_id = Some(id);
            } else if numeric_name_is(name, "capacity()") {
                capacity = initial_numeric.get(id).copied().unwrap_or(f64::MAX);
            } else if numeric_name_is(name, "max_vehicles()") {
                max_vehicles = initial_numeric.get(id).copied().unwrap_or(f64::MAX);
            }
        }

        let load_id = load_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not find exact numeric variable load()"
                .to_string()
                .into()
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not find exact numeric variable vehicles()"
                .to_string()
                .into()
        })?;

        // Demands are optional because the translator may fold them into
        // derived comparisons. Only exact full-name matches are accepted.
        let mut demands = vec![0.0; customer_count];
        for (customer, object_name) in customer_names.iter().enumerate() {
            let bare = format!("demand({object_name})");
            for (id, variable) in numeric_variables.iter().enumerate() {
                if numeric_name_is(variable.name(), &bare) {
                    demands[customer] = initial_numeric.get(id).copied().unwrap_or(0.0);
                    break;
                }
            }
        }

        let matrix_size = place_count * place_count;
        let mut direct_cost = vec![INF; matrix_size];
        let mut via_cost = vec![INF; matrix_size];
        for place in 0..place_count {
            direct_cost[place * place_count + place] = 0.0;
        }

        // Phase A: primary reconstruction from grounded operators.
        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }
            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (place_index.get(*from), place_index.get(*to))
                    {
                        direct_cost[from_id * place_count + to_id] = cost;
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (place_index.get(*from), place_index.get(*to))
                    {
                        via_cost[from_id * place_count + to_id] = cost;
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = place_index.get(*from) {
                        direct_cost[from_id * place_count + depot_index] = cost;
                    }
                }
                _ => {}
            }
        }

        // Phase B: fill only missing direct arcs from explicit travel-cost PNEs.
        for (id, variable) in numeric_variables.iter().enumerate() {
            if let Some((from, to)) = parse_travel_cost_name(variable.name()) {
                if let (Some(&from_id), Some(&to_id)) =
                    (place_index.get(&from), place_index.get(&to))
                {
                    let index = from_id * place_count + to_id;
                    if !direct_cost[index].is_finite() {
                        let value = initial_numeric.get(id).copied().unwrap_or(INF);
                        if value.is_finite() && value >= 0.0 {
                            direct_cost[index] = value;
                        }
                    }
                }
            }
        }

        // Phase C: use known depot legs to complete missing via-depot arcs.
        for from in 0..customer_count {
            for to in 0..customer_count {
                if from == to {
                    continue;
                }
                let index = from * place_count + to;
                if !via_cost[index].is_finite() {
                    let back = direct_cost[from * place_count + depot_index];
                    let out = direct_cost[depot_index * place_count + to];
                    if back.is_finite() && out.is_finite() {
                        via_cost[index] = back + out;
                    }
                }
            }
        }

        let usable_departures = (0..customer_count)
            .all(|customer| direct_cost[depot_index * place_count + customer].is_finite());
        let usable_returns = (0..customer_count)
            .all(|customer| direct_cost[customer * place_count + depot_index].is_finite());
        if !usable_departures || !usable_returns {
            return Err(
                "llm_001 could not reconstruct all depot departure and return costs"
                    .to_string()
                    .into(),
            );
        }

        // Pre-sort directed customer adjacency once. No symmetry is assumed.
        let mut nearest_from = Vec::with_capacity(place_count);
        for from in 0..place_count {
            let mut neighbors: Vec<usize> = (0..customer_count)
                .filter(|&to| to != from && direct_cost[from * place_count + to].is_finite())
                .collect();
            neighbors.sort_by(|&left, &right| {
                direct_cost[from * place_count + left]
                    .partial_cmp(&direct_cost[from * place_count + right])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_from.push(neighbors);
        }

        Ok(Self {
            customer_count,
            depot_index,
            visited_facts,
            location_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            direct_cost,
            via_cost,
            nearest_from,
            scratch_numeric: RefCell::new(vec![0.0; numeric_variables.len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; customer_count]),
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

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        let current_location = self
            .location_facts
            .iter()
            .position(|fact| fact.holds(&propositional_values))
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "llm_001 could not identify the current location".to_string(),
                )
            })?;

        let load = numeric_values.get(self.load_id).copied().unwrap_or(0.0).max(0.0);
        let vehicles = numeric_values
            .get(self.vehicles_id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        for customer in 0..self.customer_count {
            let pending = !self.visited_facts[customer].holds(&propositional_values);
            unvisited[customer] = pending;
            if pending {
                remaining_count += 1;
                remaining_demand += self.demands[customer].max(0.0);
            }
        }

        // All customers have been served: only the mandatory final return remains.
        if remaining_count == 0 {
            if current_location == self.depot_index {
                return Ok(0.0);
            }
            let return_cost = self.direct(current_location, self.depot_index);
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                LARGE_PENALTY
            });
        }

        let capacity_known = self.capacity.is_finite() && self.capacity < f64::MAX;
        let fleet_known = self.max_vehicles.is_finite() && self.max_vehicles < f64::MAX;

        if capacity_known {
            for customer in 0..self.customer_count {
                if unvisited[customer] && self.demands[customer] > self.capacity + 1.0e-9 {
                    return Ok(LARGE_PENALTY);
                }
            }
        }

        let residual_capacity = if capacity_known {
            (self.capacity - load).max(0.0)
        } else {
            f64::MAX
        };
        let remaining_fleet = if fleet_known {
            (self.max_vehicles - vehicles).max(0.0)
        } else {
            f64::MAX
        };

        // Resource dead-end test: demand must fit in the current residual load
        // plus the capacity of every still-available vehicle.
        if capacity_known && fleet_known {
            let available_capacity = residual_capacity + remaining_fleet * self.capacity;
            if remaining_demand > available_capacity + 1.0e-9 {
                return Ok(LARGE_PENALTY);
            }
        }

        let additional_demand = if capacity_known {
            (remaining_demand - residual_capacity).max(0.0)
        } else {
            0.0
        };
        let required_extra_routes = if capacity_known && self.capacity > 0.0 {
            (additional_demand / self.capacity).ceil()
        } else {
            0.0
        };

        if fleet_known && required_extra_routes > remaining_fleet + 1.0e-9 {
            return Ok(LARGE_PENALTY);
        }

        // First transition from the actual current location. Direct visit is
        // considered only when load permits; a via-depot transition is a
        // separate alternative and requires another vehicle.
        let mut first_transition = INF;
        for customer in 0..self.customer_count {
            if !unvisited[customer] {
                continue;
            }

            let demand = self.demands[customer].max(0.0);
            let direct_allowed = current_location == self.depot_index
                || !capacity_known
                || load + demand <= self.capacity + 1.0e-9;

            if direct_allowed {
                first_transition = first_transition.min(self.direct(current_location, customer));
            }

            if current_location != self.depot_index
                && (!fleet_known || remaining_fleet >= 1.0)
            {
                first_transition = first_transition.min(self.via(current_location, customer));
            }
        }

        if !first_transition.is_finite() {
            return Ok(LARGE_PENALTY);
        }

        // Directed one-out geometric relaxation over the pending set. A fixed
        // number of pre-sorted neighbors is probed, keeping evaluation linear.
        let mut outgoing_sum = 0.0;
        for from in 0..self.customer_count {
            if !unvisited[from] {
                continue;
            }
            let mut best = INF;
            for &to in self.nearest_from[from].iter().take(NEIGHBOR_PROBES) {
                if unvisited[to] {
                    best = self.direct(from, to);
                    break;
                }
            }
            if best.is_finite() {
                outgoing_sum += best;
            }
        }

        let internal_scale = if remaining_count > 1 {
            (remaining_count - 1) as f64 / remaining_count as f64
        } else {
            0.0
        };
        let internal_routing = outgoing_sum * internal_scale;

        // Every completion must finish at some pending customer and return to
        // the depot, so include an explicit directed final-return estimate.
        let mut final_return = INF;
        let mut cheapest_departure = INF;
        for customer in 0..self.customer_count {
            if unvisited[customer] {
                final_return = final_return.min(self.direct(customer, self.depot_index));
                cheapest_departure =
                    cheapest_departure.min(self.direct(self.depot_index, customer));
            }
        }
        if !final_return.is_finite() || !cheapest_departure.is_finite() {
            return Ok(LARGE_PENALTY);
        }

        let h_routing = first_transition + internal_routing + final_return;

        // Capacity component: each unavoidable additional route introduces a
        // depot return/departure boundary. Fleet pressure makes states with
        // fewer spare vehicles less attractive when the exact fleet limit is
        // available.
        let route_boundary = final_return + cheapest_departure;
        let h_capacity = required_extra_routes * route_boundary;
        let h_vehicles = if fleet_known && required_extra_routes > 0.0 {
            0.1 * route_boundary * required_extra_routes / (remaining_fleet + 1.0)
        } else {
            0.0
        };

        let h = h_routing + h_capacity + h_vehicles;
        if h.is_finite() && h >= 0.0 {
            Ok(h)
        } else {
            Ok(LARGE_PENALTY)
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
