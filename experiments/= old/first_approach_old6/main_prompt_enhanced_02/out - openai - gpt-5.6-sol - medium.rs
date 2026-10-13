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

const MISSING_COST: f64 = f64::INFINITY;
const DEAD_END_PENALTY: f64 = 1.0e15;
const NEIGHBOR_PROBES: usize = 8;

fn parse_atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let atom = name.trim().strip_prefix("Atom ")?.trim();
    let open = atom.find('(')?;
    if &atom[..open] != predicate || !atom.ends_with(')') {
        return None;
    }
    let argument = atom[open + 1..atom.len() - 1].trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument)
    }
}

fn normalized_pne_name(name: &str) -> &str {
    let trimmed = name.trim();
    match trimmed.strip_prefix("PNE ") {
        Some(rest) => rest.trim(),
        None => trimmed,
    }
}

fn parse_function(name: &str) -> Option<(&str, Vec<&str>)> {
    let text = normalized_pne_name(name);
    let open = text.find('(')?;
    if !text.ends_with(')') {
        return None;
    }

    let function = text[..open].trim();
    if function.is_empty() {
        return None;
    }

    let arguments_text = &text[open + 1..text.len() - 1];
    if arguments_text.trim().is_empty() {
        return Some((function, Vec::new()));
    }

    let arguments: Vec<&str> = arguments_text.split(',').map(str::trim).collect();
    if arguments.iter().any(|arg| arg.is_empty()) {
        None
    } else {
        Some((function, arguments))
    }
}

fn parse_operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

fn finite_cost(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

struct LlmHeuristic {
    place_names: Vec<String>,
    loc_facts: Vec<ExplicitFact>,
    customer_place: Vec<usize>,
    visited_facts: Vec<ExplicitFact>,
    depot: usize,

    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    nearest_customers: Vec<Vec<usize>>,
    minimum_outgoing: Vec<f64>,
    minimum_boundary: f64,

    demand: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_id: usize,
    vehicles_id: usize,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover all loc and visited facts from semantic fact names. No SAS
        // variable/value convention is assumed.
        let mut loc_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        let mut visited_by_name: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(place) = parse_atom_argument(fact_name, "loc") {
                    loc_by_name.insert(place.to_string(), fact.clone());
                }
                if let Some(customer) = parse_atom_argument(fact_name, "visited") {
                    visited_by_name.insert(customer.to_string(), fact.clone());
                }
            }
        }

        if loc_by_name.is_empty() || visited_by_name.is_empty() {
            return Err(
                "llm_001 could not discover CVRP loc/visited facts"
                    .to_string()
                    .into(),
            );
        }

        let mut place_names: Vec<String> = loc_by_name.keys().cloned().collect();
        place_names.sort();
        let place_index: HashMap<String, usize> = place_names
            .iter()
            .enumerate()
            .map(|(id, name)| (name.clone(), id))
            .collect();

        let mut loc_facts = Vec::with_capacity(place_names.len());
        for name in &place_names {
            loc_facts.push(loc_by_name[name].clone());
        }

        // In this domain the depot has a loc fact but no visited fact.
        let depot_candidates: Vec<usize> = place_names
            .iter()
            .enumerate()
            .filter_map(|(id, name)| {
                if visited_by_name.contains_key(name) {
                    None
                } else {
                    Some(id)
                }
            })
            .collect();

        if depot_candidates.len() != 1 {
            return Err(
                "llm_001 expected exactly one depot-like loc place without a visited fact"
                    .to_string()
                    .into(),
            );
        }
        let depot = depot_candidates[0];

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        let mut customer_place = Vec::with_capacity(customer_names.len());
        let mut visited_facts = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let place = match place_index.get(name) {
                Some(id) => *id,
                None => {
                    return Err(
                        format!("llm_001 found visited fact for {name} but no corresponding loc fact")
                            .into(),
                    );
                }
            };
            customer_place.push(place);
            visited_facts.push(visited_by_name[name].clone());
        }

        // Resolve mandatory dynamic numeric variables once. Optional resource
        // constants are accepted only as exact parsed PNE function names.
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;
        let mut demand = vec![0.0; customer_names.len()];
        let customer_index: HashMap<String, usize> = customer_names
            .iter()
            .enumerate()
            .map(|(id, name)| (name.clone(), id))
            .collect();

        let initial_numeric = task.get_initial_numeric_state_values();
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let parsed = parse_function(variable.name());
            if let Some((function, arguments)) = parsed {
                if function == "load" && arguments.is_empty() {
                    load_id = Some(id);
                } else if function == "vehicles" && arguments.is_empty() {
                    vehicles_id = Some(id);
                } else if function == "capacity" && arguments.is_empty() {
                    if let Some(value) = initial_numeric.get(id) {
                        if value.is_finite() && *value > 0.0 {
                            capacity = *value;
                        }
                    }
                } else if function == "max_vehicles" && arguments.is_empty() {
                    if let Some(value) = initial_numeric.get(id) {
                        if value.is_finite() && *value >= 0.0 {
                            max_vehicles = *value;
                        }
                    }
                } else if function == "demand" && arguments.len() == 1 {
                    if let (Some(customer), Some(value)) =
                        (customer_index.get(arguments[0]), initial_numeric.get(id))
                    {
                        if value.is_finite() && *value >= 0.0 {
                            demand[*customer] = *value;
                        }
                    }
                }
            }
        }

        let load_id = match load_id {
            Some(id) => id,
            None => {
                return Err("llm_001 could not find exact zero-arity load() PNE"
                    .to_string()
                    .into());
            }
        };
        let vehicles_id = match vehicles_id {
            Some(id) => id,
            None => {
                return Err("llm_001 could not find exact zero-arity vehicles() PNE"
                    .to_string()
                    .into());
            }
        };

        let place_count = place_names.len();
        let mut direct_cost = vec![vec![MISSING_COST; place_count]; place_count];
        let mut via_depot_cost = vec![vec![MISSING_COST; place_count]; place_count];
        for id in 0..place_count {
            direct_cost[id][id] = 0.0;
        }

        // Phase A: grounded operators are the primary and authoritative source.
        for operator in task.get_operators() {
            let tokens = parse_operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }
            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !finite_cost(cost) {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(from), Some(to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        direct_cost[*from][*to] = direct_cost[*from][*to].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(from), Some(to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        via_depot_cost[*from][*to] =
                            via_depot_cost[*from][*to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(from) = place_index.get(tokens[1]) {
                        direct_cost[*from][depot] = direct_cost[*from][depot].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Phase B: exact syntactic parsing of travel-cost(from,to) PNEs fills
        // only missing entries. Derived-expression substrings are never used.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let Some((function, arguments)) = parse_function(variable.name()) else {
                continue;
            };
            if function != "travel-cost" || arguments.len() != 2 {
                continue;
            }
            let (Some(from), Some(to), Some(value)) = (
                place_index.get(arguments[0]),
                place_index.get(arguments[1]),
                initial_numeric.get(id),
            ) else {
                continue;
            };
            if !direct_cost[*from][*to].is_finite() && finite_cost(*value) {
                direct_cost[*from][*to] = *value;
            }
        }

        // Phase C: fill remaining via-depot transitions from directed depot
        // legs, then use those transitions as a conservative last resort for
        // missing direct arcs. Directional costs are kept directional.
        for from in 0..place_count {
            for to in 0..place_count {
                if from == to {
                    continue;
                }
                let through_depot = if direct_cost[from][depot].is_finite()
                    && direct_cost[depot][to].is_finite()
                {
                    direct_cost[from][depot] + direct_cost[depot][to]
                } else {
                    MISSING_COST
                };

                if !via_depot_cost[from][to].is_finite() && through_depot.is_finite() {
                    via_depot_cost[from][to] = through_depot;
                }
                if !direct_cost[from][to].is_finite() && through_depot.is_finite() {
                    direct_cost[from][to] = through_depot;
                }
            }
        }

        let mut usable = true;
        for &place in &customer_place {
            if !direct_cost[depot][place].is_finite()
                || !direct_cost[place][depot].is_finite()
            {
                usable = false;
                break;
            }
        }
        if !usable {
            return Err(
                "llm_001 could not reconstruct usable directed depot/customer routing costs"
                    .to_string()
                    .into(),
            );
        }

        // Dense instances with tens of customers benefit from sorting once.
        // Evaluation probes only a bounded prefix of these lists.
        let mut nearest_customers = vec![Vec::new(); customer_names.len()];
        let mut minimum_outgoing = vec![0.0; customer_names.len()];
        for customer in 0..customer_names.len() {
            let from = customer_place[customer];
            let mut neighbors: Vec<usize> = (0..customer_names.len())
                .filter(|other| *other != customer)
                .filter(|other| direct_cost[from][customer_place[*other]].is_finite())
                .collect();
            neighbors.sort_by(|a, b| {
                direct_cost[from][customer_place[*a]]
                    .total_cmp(&direct_cost[from][customer_place[*b]])
            });

            minimum_outgoing[customer] = neighbors
                .first()
                .map(|other| direct_cost[from][customer_place[*other]])
                .unwrap_or(direct_cost[from][depot]);
            nearest_customers[customer] = neighbors;
        }

        let mut minimum_boundary = MISSING_COST;
        for &from in &customer_place {
            for &to in &customer_place {
                if direct_cost[from][depot].is_finite()
                    && direct_cost[depot][to].is_finite()
                {
                    minimum_boundary = minimum_boundary
                        .min(direct_cost[from][depot] + direct_cost[depot][to]);
                }
            }
        }
        if !minimum_boundary.is_finite() {
            minimum_boundary = 0.0;
        }

        Ok(Self {
            place_names,
            loc_facts,
            customer_place,
            visited_facts,
            depot,
            direct_cost,
            via_depot_cost,
            nearest_customers,
            minimum_outgoing,
            minimum_boundary,
            demand,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; customer_names.len()]),
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
        let view = registry.view(state);

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        view.fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let load = numeric_values
            .get(self.load_id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let vehicles = numeric_values
            .get(self.vehicles_id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        drop(numeric_values);

        let current_place = self
            .loc_facts
            .iter()
            .position(|fact| fact.is_hold(view));
        let Some(current_place) = current_place else {
            return Ok(DEAD_END_PENALTY);
        };

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;
        for customer in 0..self.visited_facts.len() {
            let pending = !self.visited_facts[customer].is_hold(view);
            unvisited[customer] = pending;
            if pending {
                remaining_count += 1;
                remaining_demand += self.demand[customer];
            }
        }

        // Exact edge case: all service goals hold, so only the mandatory final
        // return-to-depot action remains.
        if remaining_count == 0 {
            if current_place == self.depot {
                return Ok(0.0);
            }
            let return_cost = self.direct_cost[current_place][self.depot];
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                DEAD_END_PENALTY
            });
        }

        let capacity_known = self.capacity.is_finite() && self.capacity < f64::MAX;
        let fleet_known = self.max_vehicles.is_finite() && self.max_vehicles < f64::MAX;

        let residual_capacity = if capacity_known {
            (self.capacity - load).max(0.0)
        } else {
            f64::MAX
        };
        let additional_vehicle_slots = if fleet_known {
            (self.max_vehicles - vehicles).max(0.0)
        } else {
            f64::MAX
        };

        // Best-effort dead-end check. If optional PNEs were folded away,
        // permissive defaults make this check inactive rather than unsoundly
        // rejecting the state.
        if capacity_known && fleet_known {
            let available_capacity = residual_capacity
                + additional_vehicle_slots * self.capacity;
            if remaining_demand > available_capacity + 1.0e-9 {
                return Ok(DEAD_END_PENALTY + remaining_demand.min(1.0e12));
            }
        }

        let via_allowed = !fleet_known || vehicles + 1.0 <= self.max_vehicles + 1.0e-9;
        let at_depot = current_place == self.depot;

        // h_routing starts with the cheapest action that respects the domain's
        // direct-visit versus visit-via-depot distinction in the current state.
        let mut first_transition = MISSING_COST;
        let mut first_customer = None;
        let mut first_is_via = false;

        for customer in 0..self.visited_facts.len() {
            if !unvisited[customer] {
                continue;
            }
            let target = self.customer_place[customer];
            let fits = !capacity_known
                || load + self.demand[customer] <= self.capacity + 1.0e-9;

            let (candidate, is_via) = if fits {
                (self.direct_cost[current_place][target], false)
            } else if !at_depot && via_allowed {
                (self.via_depot_cost[current_place][target], true)
            } else {
                (MISSING_COST, false)
            };

            if candidate < first_transition {
                first_transition = candidate;
                first_customer = Some(customer);
                first_is_via = is_via;
            }
        }

        if !first_transition.is_finite() || first_customer.is_none() {
            return Ok(DEAD_END_PENALTY);
        }

        // Directed geometric backbone over the actual unvisited set. For each
        // pending customer, inspect a bounded prefix of a pre-sorted adjacency
        // list; this captures local clusters without per-state sorting.
        let mut backbone = 0.0;
        for customer in 0..self.visited_facts.len() {
            if !unvisited[customer] || Some(customer) == first_customer {
                continue;
            }

            let from = self.customer_place[customer];
            let mut local_connection = MISSING_COST;
            for &neighbor in self.nearest_customers[customer]
                .iter()
                .take(NEIGHBOR_PROBES)
            {
                if unvisited[neighbor] && neighbor != customer {
                    local_connection = self.direct_cost[from][self.customer_place[neighbor]];
                    break;
                }
            }
            if !local_connection.is_finite() {
                local_connection = self.minimum_outgoing[customer];
            }
            if local_connection.is_finite() {
                backbone += local_connection;
            }
        }

        // Every completion must eventually return from a remaining customer to
        // the depot. Use the cheapest directed terminal leg as the estimate.
        let mut final_return = MISSING_COST;
        for customer in 0..self.visited_facts.len() {
            if unvisited[customer] {
                final_return = final_return.min(
                    self.direct_cost[self.customer_place[customer]][self.depot],
                );
            }
        }
        if !final_return.is_finite() {
            return Ok(DEAD_END_PENALTY);
        }

        let h_routing = first_transition + 0.65 * backbone + final_return;

        // Capacity component: estimate how many additional route boundaries
        // are forced by total remaining demand after using residual load.
        let mut required_reloads = 0.0;
        if capacity_known && self.capacity > 0.0 {
            let demand_after_current_route =
                (remaining_demand - residual_capacity).max(0.0);
            required_reloads = (demand_after_current_route / self.capacity).ceil();
        }

        // If the selected first transition already goes via the depot, its
        // boundary cost is already included in h_routing.
        let unpaid_reloads = if first_is_via {
            (required_reloads - 1.0).max(0.0)
        } else {
            required_reloads
        };
        let h_capacity = 0.70 * unpaid_reloads * self.minimum_boundary;

        // Separate fleet-pressure term distinguishes states with the same load
        // and customers but different numbers of vehicles already consumed.
        let h_vehicles = if fleet_known && required_reloads > 0.0 {
            if additional_vehicle_slots + 1.0e-9 < required_reloads {
                drop(unvisited);
                return Ok(DEAD_END_PENALTY + remaining_demand.min(1.0e12));
            }
            let pressure = required_reloads / additional_vehicle_slots.max(1.0);
            0.10 * pressure * self.minimum_boundary
        } else {
            0.0
        };

        drop(unvisited);

        let h = h_routing + h_capacity + h_vehicles;
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
                .into(),
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
