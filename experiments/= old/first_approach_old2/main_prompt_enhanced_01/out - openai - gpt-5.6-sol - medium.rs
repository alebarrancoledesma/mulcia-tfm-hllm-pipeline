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

const LARGE_PENALTY: f64 = 1.0e15;
const MAX_NORMAL_HEURISTIC: f64 = 1.0e14;
const EPSILON: f64 = 1.0e-9;

#[derive(Clone, Copy)]
struct FactSlot {
    variable: usize,
    value: usize,
}

struct CustomerInfo {
    place: usize,
    visited: FactSlot,
    demand: f64,
}

struct LlmHeuristic {
    customers: Vec<CustomerInfo>,
    loc_slots: Vec<FactSlot>,
    depot: usize,

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,

    // Directed direct-travel costs between all places. Customer-to-depot
    // entries come primarily from return-to-depot operators.
    direct_cost: Vec<f64>,

    // Directed costs of visit-via-depot transitions. Entries involving the
    // depot itself are unused.
    via_cost: Vec<f64>,

    // Cheapest static customer-to-customer continuation for each customer.
    min_internal_out: Vec<f64>,

    // Cheapest estimated premium for replacing an ordinary direct customer
    // transition with a visit-via-depot transition.
    route_break_premium: f64,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

fn normalized_numeric_name(raw: &str) -> &str {
    let trimmed = raw.trim();
    trimmed.strip_prefix("PNE ").unwrap_or(trimmed).trim()
}

fn parse_unary_call<'a>(text: &'a str, function: &str) -> Option<&'a str> {
    let prefix = format!("{}(", function);
    if !text.starts_with(&prefix) || !text.ends_with(')') {
        return None;
    }

    let argument = text[prefix.len()..text.len() - 1].trim();
    if argument.is_empty() || argument.contains(',') || argument.contains('(') {
        None
    } else {
        Some(argument)
    }
}

fn parse_binary_call<'a>(text: &'a str, function: &str) -> Option<(&'a str, &'a str)> {
    let prefix = format!("{}(", function);
    if !text.starts_with(&prefix) || !text.ends_with(')') {
        return None;
    }

    let arguments = &text[prefix.len()..text.len() - 1];
    let mut pieces = arguments.split(',');
    let first = pieces.next()?.trim();
    let second = pieces.next()?.trim();

    if pieces.next().is_some()
        || first.is_empty()
        || second.is_empty()
        || first.contains('(')
        || second.contains('(')
    {
        None
    } else {
        Some((first, second))
    }
}

fn parse_positive_fact(text: &str, predicate: &str) -> Option<String> {
    let prefix = format!("Atom {}(", predicate);
    if !text.starts_with(&prefix) || !text.ends_with(')') {
        return None;
    }

    let argument = text[prefix.len()..text.len() - 1].trim();
    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.to_string())
    }
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

fn is_usable_cost(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // --------------------------------------------------------------------
        // Discover positive loc and visited facts from semantic fact names.
        // Propositional variable names are deliberately not used.
        // --------------------------------------------------------------------
        let mut loc_by_name: HashMap<String, FactSlot> = HashMap::new();
        let mut visited_by_name: HashMap<String, FactSlot> = HashMap::new();

        for (variable, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(variable, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(place) = parse_positive_fact(&fact_name, "loc") {
                    loc_by_name.insert(place, FactSlot { variable, value });
                }
                if let Some(customer) = parse_positive_fact(&fact_name, "visited") {
                    visited_by_name.insert(customer, FactSlot { variable, value });
                }
            }
        }

        if loc_by_name.is_empty() || visited_by_name.is_empty() {
            return Err(
                "llm_001 could not discover positive loc/visited facts from semantic fact names"
                    .to_string()
                    .into(),
            );
        }

        let mut place_names: Vec<String> = loc_by_name.keys().cloned().collect();
        place_names.sort();

        let mut place_index: HashMap<String, usize> = HashMap::new();
        let mut loc_slots = Vec::with_capacity(place_names.len());
        for (index, name) in place_names.iter().enumerate() {
            place_index.insert(name.clone(), index);
            loc_slots.push(*loc_by_name.get(name).unwrap());
        }

        let depot_candidates: Vec<usize> = place_names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                if visited_by_name.contains_key(name) {
                    None
                } else {
                    Some(index)
                }
            })
            .collect();

        if depot_candidates.len() != 1 {
            return Err(
                "llm_001 expected exactly one loc place without a corresponding visited fact"
                    .to_string()
                    .into(),
            );
        }
        let depot = depot_candidates[0];

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        // --------------------------------------------------------------------
        // Discover numeric IDs once. Static resource values and demands are
        // read from simple PNEs when available, with expression-name fallback
        // below for compiled numeric tasks.
        // --------------------------------------------------------------------
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut exact_demands: HashMap<String, f64> = HashMap::new();

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let normalized = normalized_numeric_name(var.name());
            let value = initial_numeric[id];

            match normalized {
                "load()" => load_id = Some(id),
                "vehicles()" => vehicles_id = Some(id),
                "capacity()" => capacity = Some(value),
                "max_vehicles()" | "max-vehicles()" => max_vehicles = Some(value),
                _ => {
                    if let Some(customer) = parse_unary_call(normalized, "demand") {
                        if is_usable_cost(value) {
                            exact_demands.insert(customer.to_string(), value);
                        }
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not discover numeric variable PNE load()"
                .to_string()
                .into()
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not discover numeric variable PNE vehicles()"
                .to_string()
                .into()
        })?;

        // Some compilations expose static PNEs only inside expression names.
        // Prefer expressions that do not mention another demand for global
        // constants, and load+demand expressions for missing demands because
        // initial load is normally zero in this domain family.
        if capacity.is_none() || max_vehicles.is_none() {
            for (id, var) in task.numeric_variables().iter().enumerate() {
                let name = var.name();
                let value = initial_numeric[id];
                if !value.is_finite() {
                    continue;
                }

                if capacity.is_none()
                    && name.contains("capacity()")
                    && !name.contains("demand(")
                    && value > 0.0
                {
                    capacity = Some(value);
                }
                if max_vehicles.is_none()
                    && (name.contains("max_vehicles()") || name.contains("max-vehicles()"))
                    && value > 0.0
                {
                    max_vehicles = Some(value);
                }
            }
        }

        let capacity = capacity.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not recover capacity() from the numeric task"
                .to_string()
                .into()
        })?;
        let max_vehicles = max_vehicles.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not recover max_vehicles() from the numeric task"
                .to_string()
                .into()
        })?;

        if !capacity.is_finite() || capacity <= 0.0
            || !max_vehicles.is_finite() || max_vehicles < 1.0
        {
            return Err(
                "llm_001 found invalid capacity or max_vehicles values"
                    .to_string()
                    .into(),
            );
        }

        let mut customers = Vec::with_capacity(customer_names.len());
        for customer_name in &customer_names {
            let place = *place_index.get(customer_name).ok_or_else(|| -> HeuristicBuildError {
                format!("llm_001 found visited({}) but no corresponding loc fact", customer_name)
                    .into()
            })?;

            let mut demand = exact_demands.get(customer_name).copied();
            if demand.is_none() {
                let demand_pattern = format!("demand({})", customer_name);
                for (id, var) in task.numeric_variables().iter().enumerate() {
                    let name = var.name();
                    let value = initial_numeric[id];
                    if name.contains(&demand_pattern)
                        && name.contains("load()")
                        && value.is_finite()
                        && value >= 0.0
                    {
                        demand = Some(value);
                        break;
                    }
                }
            }

            let demand = demand.ok_or_else(|| -> HeuristicBuildError {
                format!("llm_001 could not recover demand for customer {}", customer_name)
                    .into()
            })?;

            customers.push(CustomerInfo {
                place,
                visited: *visited_by_name.get(customer_name).unwrap(),
                demand,
            });
        }

        // --------------------------------------------------------------------
        // Phase A: reconstruct directed travel costs primarily from grounded
        // operators and their metric costs.
        // --------------------------------------------------------------------
        let place_count = place_names.len();
        let matrix_size = place_count * place_count;
        let mut direct_cost = vec![f64::INFINITY; matrix_size];
        let mut via_cost = vec![f64::INFINITY; matrix_size];
        direct_cost[depot * place_count + depot] = 0.0;

        for op in task.get_operators() {
            let op_name = op.name().to_string();
            let tokens = operator_tokens(&op_name);
            let cost = metric_operator_cost_from_initial_values(task, op);
            if !is_usable_cost(cost) {
                continue;
            }

            match tokens.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (place_index.get(*from), place_index.get(*to))
                    {
                        let slot = from_id * place_count + to_id;
                        direct_cost[slot] = direct_cost[slot].min(cost);
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_id), Some(&to_id)) =
                        (place_index.get(*from), place_index.get(*to))
                    {
                        let slot = from_id * place_count + to_id;
                        via_cost[slot] = via_cost[slot].min(cost);
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_id) = place_index.get(*from) {
                        let slot = from_id * place_count + depot;
                        direct_cost[slot] = direct_cost[slot].min(cost);
                    }
                }
                _ => {}
            }
        }

        // --------------------------------------------------------------------
        // Phase B: fill still-missing direct arcs from simple travel-cost PNEs.
        // This is a fallback only; grounded operators above are primary.
        // --------------------------------------------------------------------
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let normalized = normalized_numeric_name(var.name());
            if let Some((from, to)) = parse_binary_call(normalized, "travel-cost") {
                if let (Some(&from_id), Some(&to_id)) =
                    (place_index.get(from), place_index.get(to))
                {
                    let cost = initial_numeric[id];
                    let slot = from_id * place_count + to_id;
                    if !direct_cost[slot].is_finite() && is_usable_cost(cost) {
                        direct_cost[slot] = cost;
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Phase C: derive exact via-depot costs where both directed depot legs
        // are known. A missing direct customer arc may conservatively use its
        // extracted via-depot cost as a last resort; symmetry is never assumed.
        // --------------------------------------------------------------------
        for from_customer in &customers {
            for to_customer in &customers {
                if from_customer.place == to_customer.place {
                    continue;
                }

                let from = from_customer.place;
                let to = to_customer.place;
                let via_slot = from * place_count + to;
                let return_leg = direct_cost[from * place_count + depot];
                let outbound_leg = direct_cost[depot * place_count + to];

                if !via_cost[via_slot].is_finite()
                    && return_leg.is_finite()
                    && outbound_leg.is_finite()
                {
                    via_cost[via_slot] = return_leg + outbound_leg;
                }

                let direct_slot = from * place_count + to;
                if !direct_cost[direct_slot].is_finite() && via_cost[via_slot].is_finite() {
                    direct_cost[direct_slot] = via_cost[via_slot];
                }
            }
        }

        // Validate the directional depot legs and customer transitions needed
        // by the heuristic. Silently inventing arbitrary distances would make
        // the heuristic misleading, so unusable matrices fail construction.
        for customer in &customers {
            let outbound = direct_cost[depot * place_count + customer.place];
            let inbound = direct_cost[customer.place * place_count + depot];
            if !outbound.is_finite() || !inbound.is_finite() {
                return Err(
                    "llm_001 could not reconstruct all directed depot/customer travel costs"
                        .to_string()
                        .into(),
                );
            }
        }

        if customers.len() > 1 {
            for from_customer in &customers {
                for to_customer in &customers {
                    if from_customer.place == to_customer.place {
                        continue;
                    }
                    let slot = from_customer.place * place_count + to_customer.place;
                    if !direct_cost[slot].is_finite() || !via_cost[slot].is_finite() {
                        return Err(
                            "llm_001 could not reconstruct a usable directed customer cost matrix"
                                .to_string()
                                .into(),
                        );
                    }
                }
            }
        }

        // Static routing summaries used by the O(n) evaluation.
        let mut min_internal_out = vec![0.0; customers.len()];
        let mut global_min_direct = f64::INFINITY;
        let mut global_min_via = f64::INFINITY;

        for (from_index, from_customer) in customers.iter().enumerate() {
            let mut best = f64::INFINITY;
            for to_customer in &customers {
                if from_customer.place == to_customer.place {
                    continue;
                }
                let slot = from_customer.place * place_count + to_customer.place;
                best = best.min(direct_cost[slot]);
                global_min_direct = global_min_direct.min(direct_cost[slot]);
                global_min_via = global_min_via.min(via_cost[slot]);
            }
            min_internal_out[from_index] = if best.is_finite() { best } else { 0.0 };
        }

        let route_break_premium = if global_min_direct.is_finite() && global_min_via.is_finite() {
            (global_min_via - global_min_direct).max(0.0)
        } else {
            0.0
        };

        Ok(Self {
            customers,
            loc_slots,
            depot,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            direct_cost,
            via_cost,
            min_internal_out,
            route_break_premium,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; customer_names.len()]),
        })
    }

    fn matrix_width(&self) -> usize {
        self.loc_slots.len()
    }

    fn direct(&self, from: usize, to: usize) -> f64 {
        self.direct_cost[from * self.matrix_width() + to]
    }

    fn via(&self, from: usize, to: usize) -> f64 {
        self.via_cost[from * self.matrix_width() + to]
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
        registry
            .view(state)
            .fill_numeric(&mut numeric_values);

        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        let load = numeric_values[self.load_id];
        let vehicles = numeric_values[self.vehicles_id];
        if !load.is_finite() || !vehicles.is_finite() {
            return Ok(LARGE_PENALTY);
        }

        let mut current_place = None;
        for (place, slot) in self.loc_slots.iter().enumerate() {
            if propositional_values[slot.variable] == slot.value {
                current_place = Some(place);
                break;
            }
        }
        let current_place = match current_place {
            Some(place) => place,
            None => return Ok(LARGE_PENALTY),
        };

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        let mut remaining_count = 0usize;
        let mut remaining_demand = 0.0;

        for (index, customer) in self.customers.iter().enumerate() {
            let is_visited = propositional_values[customer.visited.variable]
                == customer.visited.value;
            unvisited[index] = !is_visited;
            if !is_visited {
                remaining_count += 1;
                remaining_demand += customer.demand;
            }
        }

        // All service goals are complete, but the goal also requires the
        // vehicle to be at the depot.
        if remaining_count == 0 {
            if current_place == self.depot {
                return Ok(0.0);
            }
            let return_cost = self.direct(current_place, self.depot);
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                LARGE_PENALTY
            });
        }

        // --------------------------------------------------------------------
        // Resource infeasibility and capacity lower bound.
        // Current residual capacity belongs to the active vehicle; each
        // remaining vehicle supplies one full capacity after a via-depot move.
        // --------------------------------------------------------------------
        let residual_capacity = (self.capacity - load).max(0.0);
        let remaining_vehicles = (self.max_vehicles - vehicles).max(0.0);
        let available_capacity = residual_capacity + remaining_vehicles * self.capacity;

        if remaining_demand > available_capacity + EPSILON {
            let deficit = remaining_demand - available_capacity;
            return Ok((LARGE_PENALTY + deficit.max(0.0)).min(f64::MAX / 4.0));
        }

        let demand_after_current_vehicle = (remaining_demand - residual_capacity).max(0.0);
        let required_new_vehicles = if demand_after_current_vehicle <= EPSILON {
            0.0
        } else {
            (demand_after_current_vehicle / self.capacity).ceil()
        };

        if required_new_vehicles > remaining_vehicles + EPSILON {
            return Ok(LARGE_PENALTY);
        }

        // --------------------------------------------------------------------
        // h_entry: cheapest currently legal first transition. This explicitly
        // distinguishes direct visit from visit-via-depot semantics.
        // --------------------------------------------------------------------
        let mut entry_cost = f64::INFINITY;
        for (index, customer) in self.customers.iter().enumerate() {
            if !unvisited[index] {
                continue;
            }

            if load + customer.demand <= self.capacity + EPSILON {
                entry_cost = entry_cost.min(self.direct(current_place, customer.place));
            }

            // visit-via-depot is grounded only from a customer and consumes a
            // new vehicle. It resets load to the selected customer's demand.
            if current_place != self.depot && vehicles + 1.0 <= self.max_vehicles + EPSILON {
                entry_cost = entry_cost.min(self.via(current_place, customer.place));
            }
        }

        if !entry_cost.is_finite() {
            return Ok(LARGE_PENALTY);
        }

        // --------------------------------------------------------------------
        // h_backbone: inexpensive geometric routing estimate. Each unvisited
        // customer contributes its precomputed cheapest directed continuation.
        // Multiplication by (m-1)/m models approximately m-1 internal arcs
        // without sorting or constructing an MST at every state.
        // --------------------------------------------------------------------
        let mut continuation_sum = 0.0;
        let mut final_return_cost = f64::INFINITY;
        for (index, customer) in self.customers.iter().enumerate() {
            if unvisited[index] {
                continuation_sum += self.min_internal_out[index];
                final_return_cost = final_return_cost.min(self.direct(customer.place, self.depot));
            }
        }

        if !final_return_cost.is_finite() {
            return Ok(LARGE_PENALTY);
        }

        let internal_factor = if remaining_count > 1 {
            (remaining_count - 1) as f64 / remaining_count as f64
        } else {
            0.0
        };

        let h_routing = entry_cost
            + internal_factor * continuation_sum
            + final_return_cost;

        // Each required future vehicle change replaces an ordinary customer
        // transition with a depot-return plus depot-outbound transition.
        let h_capacity = required_new_vehicles * self.route_break_premium;

        // Mild fleet-scarcity term: among states with the same route count,
        // prefer those retaining more fleet slack. It is scaled by the same
        // task-derived route-break premium rather than an instance constant.
        let fleet_slack = (remaining_vehicles - required_new_vehicles).max(0.0);
        let h_vehicles = if required_new_vehicles > 0.0 {
            self.route_break_premium / (1.0 + fleet_slack)
        } else {
            0.0
        };

        let h = (h_routing + h_capacity + h_vehicles)
            .max(0.0)
            .min(MAX_NORMAL_HEURISTIC);

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
