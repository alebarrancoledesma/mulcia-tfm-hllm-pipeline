use std::cell::RefCell;
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
const DEAD_END_PENALTY: f64 = 1.0e15;
const EPSILON: f64 = 1.0e-9;

#[derive(Clone, Copy)]
struct FactKey {
    variable: usize,
    value: usize,
}

struct LlmHeuristic {
    // Location 0 is the depot; customer i is location i + 1.
    customer_names: Vec<String>,
    visited_facts: Vec<FactKey>,
    location_facts: Vec<FactKey>,

    load_id: usize,
    vehicles_id: usize,
    capacity_id: Option<usize>,
    max_vehicles_id: Option<usize>,

    demands: Vec<f64>,
    direct_costs: Vec<f64>,
    via_depot_costs: Vec<f64>,
    local_link_costs: Vec<f64>,
    reload_unit_cost: f64,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn canonical_numeric_name(name: &str) -> &str {
        let trimmed = name.trim();
        trimmed.strip_prefix("PNE ").unwrap_or(trimmed).trim()
    }

    fn atom_argument(name: &str, predicate: &str) -> Option<String> {
        let text = name.trim();
        let atom = text.strip_prefix("Atom ")?;
        let prefix = format!("{}(", predicate);
        let argument = atom.strip_prefix(&prefix)?.strip_suffix(')')?.trim();
        if argument.is_empty() || argument.contains(',') {
            None
        } else {
            Some(argument.to_string())
        }
    }

    fn parse_operator_name(name: &str) -> Vec<&str> {
        name.trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split_whitespace()
            .collect()
    }

    fn parse_travel_cost_name(name: &str) -> Option<(String, String)> {
        let canonical = Self::canonical_numeric_name(name);
        let arguments = canonical
            .strip_prefix("travel-cost(")?
            .strip_suffix(')')?;

        let mut parts = arguments.split(',').map(str::trim);
        let from = parts.next()?;
        let to = parts.next()?;
        if parts.next().is_some() || from.is_empty() || to.is_empty() {
            return None;
        }
        Some((from.to_string(), to.to_string()))
    }

    fn matrix_index(location_count: usize, from: usize, to: usize) -> usize {
        from * location_count + to
    }

    fn valid_cost(cost: f64) -> bool {
        cost.is_finite() && cost >= 0.0
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover all positive loc/visited facts from their semantic fact names.
        let mut loc_by_object: HashMap<String, FactKey> = HashMap::new();
        let mut visited_by_object: HashMap<String, FactKey> = HashMap::new();

        for variable in 0..task.variables().len() {
            for value in 0..task.variables()[variable].domain_size() {
                let fact = ExplicitFact::propositional(variable, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = Self::atom_argument(fact_name, "loc") {
                    loc_by_object.insert(object, FactKey { variable, value });
                }
                if let Some(object) = Self::atom_argument(fact_name, "visited") {
                    visited_by_object.insert(object, FactKey { variable, value });
                }
            }
        }

        if visited_by_object.is_empty() {
            return Err(
                "llm_001 could not discover any positive visited(customer) facts"
                    .to_string()
                    .into(),
            );
        }

        let mut depot_candidates: Vec<String> = loc_by_object
            .keys()
            .filter(|name| !visited_by_object.contains_key(*name))
            .cloned()
            .collect();
        depot_candidates.sort();

        if depot_candidates.len() != 1 {
            return Err(format!(
                "llm_001 expected one depot location, but discovered {} candidates",
                depot_candidates.len()
            ).into());
        }
        let depot_name = depot_candidates.pop().unwrap();

        let mut customer_names: Vec<String> = visited_by_object.keys().cloned().collect();
        customer_names.sort();

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        let mut location_facts = Vec::with_capacity(customer_names.len() + 1);
        location_facts.push(*loc_by_object.get(&depot_name).ok_or_else(|| {
            "llm_001 discovered a depot without a loc fact".to_string()
        })?);

        for customer in &customer_names {
            visited_facts.push(*visited_by_object.get(customer).ok_or_else(|| {
                format!("llm_001 lost the visited fact for {customer}")
            })?);
            location_facts.push(*loc_by_object.get(customer).ok_or_else(|| {
                format!("llm_001 found visited({customer}) but no loc({customer}) fact")
            })?);
        }

        // Resolve mandatory dynamic numeric variables and optional resource constants.
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity_id = None;
        let mut max_vehicles_id = None;
        let mut demand_ids: HashMap<String, usize> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = Self::canonical_numeric_name(variable.name());
            match name {
                "load()" => load_id = Some(id),
                "vehicles()" => vehicles_id = Some(id),
                "capacity()" => capacity_id = Some(id),
                "max_vehicles()" => max_vehicles_id = Some(id),
                _ => {
                    if let Some(argument) = name
                        .strip_prefix("demand(")
                        .and_then(|rest| rest.strip_suffix(')'))
                    {
                        let argument = argument.trim();
                        if !argument.is_empty() && !argument.contains(',') {
                            demand_ids.insert(argument.to_string(), id);
                        }
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not find the exact numeric variable load()".to_string()
            )
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            HeuristicBuildError::from(
                "llm_001 could not find the exact numeric variable vehicles()".to_string()
            )
        })?;

        let initial_numeric = task.get_initial_numeric_state_values();
        let demands: Vec<f64> = customer_names
            .iter()
            .map(|customer| {
                demand_ids
                    .get(customer)
                    .and_then(|id| initial_numeric.get(*id))
                    .copied()
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .unwrap_or(0.0)
            })
            .collect();

        let customer_count = customer_names.len();
        let location_count = customer_count + 1;
        let mut location_index: HashMap<String, usize> = HashMap::new();
        location_index.insert(depot_name.clone(), 0);
        for (customer, name) in customer_names.iter().enumerate() {
            location_index.insert(name.clone(), customer + 1);
        }

        let mut direct_costs = vec![INF; location_count * location_count];
        let mut via_depot_costs = vec![INF; location_count * location_count];
        for location in 0..location_count {
            direct_costs[Self::matrix_index(location_count, location, location)] = 0.0;
        }

        // Phase A: reconstruct routing costs primarily from grounded operators.
        for operator in task.get_operators() {
            let tokens = Self::parse_operator_name(operator.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !Self::valid_cost(cost) {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (location_index.get(*from), location_index.get(*to))
                    {
                        direct_costs[Self::matrix_index(location_count, from_id, to_id)] = cost;
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (location_index.get(*from), location_index.get(*to))
                    {
                        via_depot_costs[
                            Self::matrix_index(location_count, from_id, to_id)
                        ] = cost;
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = location_index.get(*from) {
                        direct_costs[Self::matrix_index(location_count, from_id, 0)] = cost;
                    }
                }
                _ => {}
            }
        }

        // Phase B: fill only missing direct arcs from exact travel-cost(...) PNEs.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let Some((from, to)) = Self::parse_travel_cost_name(variable.name()) else {
                continue;
            };
            let (Some(&from_id), Some(&to_id)) =
                (location_index.get(&from), location_index.get(&to))
            else {
                continue;
            };

            let matrix_id = Self::matrix_index(location_count, from_id, to_id);
            if !direct_costs[matrix_id].is_finite() {
                if let Some(&cost) = initial_numeric.get(id) {
                    if Self::valid_cost(cost) {
                        direct_costs[matrix_id] = cost;
                    }
                }
            }
        }

        // Depot legs are indispensable: they are needed for route starts, reloads,
        // and the mandatory final return.
        for customer in 0..customer_count {
            let location = customer + 1;
            let outbound = direct_costs[Self::matrix_index(location_count, 0, location)];
            let inbound = direct_costs[Self::matrix_index(location_count, location, 0)];
            if !Self::valid_cost(outbound) || !Self::valid_cost(inbound) {
                return Err(format!(
                    "llm_001 could not reconstruct both depot legs for customer index {customer}"
                ).into());
            }
        }

        // Phase C: conservatively fill missing customer arcs through the depot.
        // Also reconstruct missing via-depot costs from directed depot legs.
        for from in 1..location_count {
            for to in 1..location_count {
                if from == to {
                    continue;
                }
                let matrix_id = Self::matrix_index(location_count, from, to);
                let through_depot =
                    direct_costs[Self::matrix_index(location_count, from, 0)]
                    + direct_costs[Self::matrix_index(location_count, 0, to)];

                if !direct_costs[matrix_id].is_finite() {
                    direct_costs[matrix_id] = through_depot;
                }
                if !via_depot_costs[matrix_id].is_finite() {
                    via_depot_costs[matrix_id] = through_depot;
                }
            }
        }

        if direct_costs.iter().any(|cost| !cost.is_finite()) {
            return Err(
                "llm_001 produced an unusable directed routing matrix"
                    .to_string()
                    .into(),
            );
        }

        // Static geometric link estimate for each customer. Direction is preserved:
        // no symmetry assumption is made.
        let mut local_link_costs = vec![0.0; customer_count];
        for customer in 0..customer_count {
            let from = customer + 1;
            let mut nearest = INF;
            for other in 0..customer_count {
                if customer == other {
                    continue;
                }
                let to = other + 1;
                nearest = nearest.min(
                    direct_costs[Self::matrix_index(location_count, from, to)]
                );
            }
            local_link_costs[customer] = if nearest.is_finite() { nearest } else { 0.0 };
        }

        // Typical cost of creating another route. Prefer the observed difference
        // between a via-depot action and its direct counterpart.
        let mut reload_sum = 0.0;
        let mut reload_samples = 0usize;
        for from in 1..location_count {
            for to in 1..location_count {
                if from == to {
                    continue;
                }
                let matrix_id = Self::matrix_index(location_count, from, to);
                let extra = via_depot_costs[matrix_id] - direct_costs[matrix_id];
                if extra.is_finite() && extra > EPSILON {
                    reload_sum += extra;
                    reload_samples += 1;
                }
            }
        }

        let reload_unit_cost = if reload_samples > 0 {
            reload_sum / reload_samples as f64
        } else {
            let mut round_trip_sum = 0.0;
            for customer in 0..customer_count {
                let location = customer + 1;
                round_trip_sum +=
                    direct_costs[Self::matrix_index(location_count, 0, location)]
                    + direct_costs[Self::matrix_index(location_count, location, 0)];
            }
            if customer_count > 0 {
                0.5 * round_trip_sum / customer_count as f64
            } else {
                0.0
            }
        };

        Ok(Self {
            customer_names,
            visited_facts,
            location_facts,
            load_id,
            vehicles_id,
            capacity_id,
            max_vehicles_id,
            demands,
            direct_costs,
            via_depot_costs,
            local_link_costs,
            reload_unit_cost,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
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

        let _task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let (load, vehicles, capacity, max_vehicles) = {
            let mut numeric_values = self.scratch_numeric.borrow_mut();
            registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

            let load = numeric_values[self.load_id];
            let vehicles = numeric_values[self.vehicles_id];
            let capacity = self.capacity_id
                .and_then(|id| numeric_values.get(id).copied())
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(f64::MAX);
            let max_vehicles = self.max_vehicles_id
                .and_then(|id| numeric_values.get(id).copied())
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(f64::MAX);
            (load.max(0.0), vehicles.max(0.0), capacity, max_vehicles)
        };

        let current_location = {
            let mut propositional_values = self.scratch_propositional.borrow_mut();
            state.fill_state(registry, &mut propositional_values);

            let mut current = None;
            for (location, fact) in self.location_facts.iter().enumerate() {
                if propositional_values[fact.variable] == fact.value {
                    current = Some(location);
                    break;
                }
            }

            let mut unvisited = self.scratch_unvisited.borrow_mut();
            for (customer, fact) in self.visited_facts.iter().enumerate() {
                unvisited[customer] = propositional_values[fact.variable] != fact.value;
            }

            current.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "llm_001 could not identify the current location".to_string()
                )
            })?
        };

        let customer_count = self.customer_names.len();
        let location_count = customer_count + 1;
        let unvisited = self.scratch_unvisited.borrow();

        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        let mut oversized_customer = false;
        for customer in 0..customer_count {
            if unvisited[customer] {
                remaining_count += 1;
                remaining_demand += self.demands[customer];
                if capacity.is_finite()
                    && capacity < f64::MAX
                    && self.demands[customer] > capacity + EPSILON
                {
                    oversized_customer = true;
                }
            }
        }

        // All customers are served: the only remaining routing action is the
        // mandatory return to the depot.
        if remaining_count == 0 {
            if current_location == 0 {
                return Ok(0.0);
            }
            return Ok(self.direct_costs[
                Self::matrix_index(location_count, current_location, 0)
            ]);
        }

        // Best-effort resource dead-end test. If optional constants/demands were
        // folded away by translation, their permissive defaults disable only this
        // test while leaving the routing estimate intact.
        if oversized_customer {
            return Ok(DEAD_END_PENALTY);
        }

        let finite_capacity = capacity.is_finite() && capacity < f64::MAX && capacity > EPSILON;
        let finite_fleet = max_vehicles.is_finite() && max_vehicles < f64::MAX;
        let residual_capacity = if finite_capacity {
            (capacity - load).max(0.0)
        } else {
            f64::MAX
        };
        let remaining_vehicles = if finite_fleet {
            (max_vehicles - vehicles).max(0.0)
        } else {
            f64::MAX
        };

        if finite_capacity && finite_fleet {
            let total_available = residual_capacity + remaining_vehicles * capacity;
            if remaining_demand > total_available + EPSILON {
                return Ok(DEAD_END_PENALTY);
            }
        }

        // Exact cheapest feasible first transition. A direct visit is available
        // only if the customer fits; a via-depot visit additionally consumes a
        // vehicle and is unavailable while currently at the depot.
        let can_open_vehicle = !finite_fleet || vehicles + 1.0 <= max_vehicles + EPSILON;
        let mut entry_cost = INF;
        let mut entry_uses_depot = false;

        for customer in 0..customer_count {
            if !unvisited[customer] {
                continue;
            }
            let to = customer + 1;
            let fits_current_route = !finite_capacity
                || load + self.demands[customer] <= capacity + EPSILON;

            if fits_current_route {
                let direct = self.direct_costs[
                    Self::matrix_index(location_count, current_location, to)
                ];
                if direct < entry_cost {
                    entry_cost = direct;
                    entry_uses_depot = false;
                }
            }

            if current_location != 0 && can_open_vehicle {
                let via = self.via_depot_costs[
                    Self::matrix_index(location_count, current_location, to)
                ];
                if via < entry_cost {
                    entry_cost = via;
                    entry_uses_depot = true;
                }
            }
        }

        if !entry_cost.is_finite() {
            return Ok(DEAD_END_PENALTY);
        }

        // Geometric routing component: first feasible transition, a directed
        // nearest-link chain surrogate, and an explicit final return to the depot.
        let mut link_sum = 0.0;
        let mut largest_link = 0.0;
        let mut final_return = INF;
        for customer in 0..customer_count {
            if !unvisited[customer] {
                continue;
            }
            let link = self.local_link_costs[customer];
            link_sum += link;
            largest_link = largest_link.max(link);

            let location = customer + 1;
            final_return = final_return.min(self.direct_costs[
                Self::matrix_index(location_count, location, 0)
            ]);
        }

        // A path through n customers has only n-1 customer-to-customer links.
        let internal_links = if remaining_count > 1 {
            (link_sum - largest_link).max(0.0)
        } else {
            0.0
        };
        let h_routing = entry_cost + internal_links + final_return;

        // Capacity component: estimate how many route reloads are forced by the
        // demand not fitting in the residual capacity of the current vehicle.
        let mut forced_reloads = 0.0;
        if finite_capacity && remaining_demand > residual_capacity + EPSILON {
            forced_reloads = ((remaining_demand - residual_capacity) / capacity).ceil();
        }

        // If the selected first transition already goes via the depot, its routing
        // cost explicitly pays for one of these reloads.
        let future_reloads = if entry_uses_depot {
            (forced_reloads - 1.0).max(0.0)
        } else {
            forced_reloads
        };
        let h_capacity = future_reloads * self.reload_unit_cost;

        // Fleet/load pressure is deliberately small but distinguishes states even
        // when the translator retained only load() and vehicles().
        let load_pressure = if finite_capacity {
            (load / capacity).clamp(0.0, 2.0)
        } else {
            load / (1.0 + load)
        };
        let vehicle_pressure = if finite_fleet && max_vehicles > EPSILON {
            (vehicles / max_vehicles).clamp(0.0, 2.0)
        } else {
            vehicles / (1.0 + vehicles)
        };
        let h_resource_pressure =
            0.05 * self.reload_unit_cost * (load_pressure + vehicle_pressure);

        let h = h_routing + h_capacity + h_resource_pressure;
        if h.is_finite() && h >= 0.0 {
            Ok(h)
        } else {
            Ok(DEAD_END_PENALTY)
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
