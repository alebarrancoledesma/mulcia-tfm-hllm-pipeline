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
use std::collections::{HashMap, HashSet};


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

const LARGE_PENALTY: f64 = 1.0e15;
const INF: f64 = f64::INFINITY;

struct CustomerInfo {
    visited_fact: ExplicitFact,
    demand: f64,
}

struct Scratch {
    numeric: Vec<f64>,
    unvisited: Vec<usize>,
}

struct LlmHeuristic {
    // Place 0 is the structurally discovered depot; places 1.. are customers.
    loc_facts: Vec<ExplicitFact>,
    customers: Vec<CustomerInfo>,

    load_id: usize,
    vehicles_id: usize,

    capacity: f64,
    max_vehicles: f64,
    capacity_known: bool,
    max_vehicles_known: bool,

    // Directed costs. No symmetry is assumed.
    direct: Vec<Vec<f64>>,
    via_depot: Vec<Vec<f64>>,

    // Cheap geometric summaries derived from the directed matrices.
    min_out: Vec<f64>,
    min_via_from: Vec<f64>,
    arc_scale: f64,

    scratch: RefCell<Scratch>,
}

fn parse_atom_object(name: &str, predicate: &str) -> Option<String> {
    let trimmed = name.trim();
    let body = trimmed.strip_prefix("Atom ").unwrap_or(trimmed);
    let prefix = format!("{}(", predicate);
    let inside = body.strip_prefix(&prefix)?.strip_suffix(')')?.trim();
    if inside.is_empty() || inside.contains(',') {
        None
    } else {
        Some(inside.to_string())
    }
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

fn parse_travel_cost_name(name: &str) -> Option<(String, String)> {
    let trimmed = name.trim();
    let body = if let Some(rest) = trimmed.strip_prefix("PNE ") {
        rest.trim()
    } else {
        trimmed
    };
    let args = body.strip_prefix("travel-cost(")?.strip_suffix(')')?;
    let mut parts = args.split(',');
    let from = parts.next()?.trim();
    let to = parts.next()?.trim();
    if parts.next().is_some() || from.is_empty() || to.is_empty() {
        return None;
    }
    Some((from.to_string(), to.to_string()))
}

fn finite_cost(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover semantic propositional facts rather than relying on SAS IDs
        // or on the generic propositional variable names.
        let mut loc_by_object: HashMap<String, ExplicitFact> = HashMap::new();
        let mut visited_by_object: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some(object) = parse_atom_object(fact_name, "loc") {
                    loc_by_object.insert(object, fact.clone());
                }
                if let Some(object) = parse_atom_object(fact_name, "visited") {
                    visited_by_object.insert(object, fact);
                }
            }
        }

        if visited_by_object.is_empty() {
            return Err("llm_001 could not discover any visited(customer) facts"
                .to_string()
                .into());
        }

        // The depot is the unique place with a loc fact but no visited fact.
        let customer_names_set: HashSet<String> =
            visited_by_object.keys().cloned().collect();
        let depot_candidates: Vec<String> = loc_by_object
            .keys()
            .filter(|name| !customer_names_set.contains(*name))
            .cloned()
            .collect();

        if depot_candidates.len() != 1 {
            return Err(format!(
                "llm_001 expected one structural depot, found {}",
                depot_candidates.len()
            )
            .into());
        }
        let depot_name = depot_candidates[0].clone();

        let mut customer_names: Vec<String> = visited_by_object.keys().cloned().collect();
        customer_names.sort();

        for customer in &customer_names {
            if !loc_by_object.contains_key(customer) {
                return Err(format!(
                    "llm_001 found visited({customer}) but no corresponding loc fact"
                )
                .into());
            }
        }

        let mut place_names = Vec::with_capacity(customer_names.len() + 1);
        place_names.push(depot_name.clone());
        place_names.extend(customer_names.iter().cloned());

        let mut place_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in place_names.iter().enumerate() {
            place_index.insert(name.clone(), index);
        }

        let mut loc_facts = Vec::with_capacity(place_names.len());
        for name in &place_names {
            loc_facts.push(loc_by_object.get(name).unwrap().clone());
        }

        // Resolve load and vehicles once. Capacity, max_vehicles, and demands
        // are exact-name, best-effort signals because the translator may fold
        // them into derived expressions.
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;
        let mut capacity_known = false;
        let mut max_vehicles_known = false;
        let mut exact_numeric_values: HashMap<String, f64> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name().trim();
            if name == "load()" || name == "PNE load()" {
                load_id = Some(id);
            }
            if name == "vehicles()" || name == "PNE vehicles()" {
                vehicles_id = Some(id);
            }
            if name == "capacity()" || name == "PNE capacity()" {
                if id < initial_numeric.len() {
                    capacity = initial_numeric[id];
                    capacity_known = finite_cost(capacity) && capacity > 0.0;
                }
            }
            if name == "max_vehicles()" || name == "PNE max_vehicles()" {
                if id < initial_numeric.len() {
                    max_vehicles = initial_numeric[id];
                    max_vehicles_known = finite_cost(max_vehicles);
                }
            }
            if id < initial_numeric.len() {
                exact_numeric_values.insert(name.to_string(), initial_numeric[id]);
            }
        }

        let load_id = load_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not find the exact zero-arity load() variable"
                .to_string()
                .into()
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not find the exact zero-arity vehicles() variable"
                .to_string()
                .into()
        })?;

        let mut customers = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            let plain = format!("demand({name})");
            let prefixed = format!("PNE demand({name})");
            let demand = exact_numeric_values
                .get(&plain)
                .or_else(|| exact_numeric_values.get(&prefixed))
                .copied()
                .filter(|value| finite_cost(*value))
                .unwrap_or(0.0);
            customers.push(CustomerInfo {
                visited_fact: visited_by_object.get(name).unwrap().clone(),
                demand,
            });
        }

        let place_count = place_names.len();
        let mut direct = vec![vec![INF; place_count]; place_count];
        let mut via_depot = vec![vec![INF; place_count]; place_count];
        for i in 0..place_count {
            direct[i][i] = 0.0;
        }

        // Phase A: grounded operators are the primary source of routing costs.
        for operator in task.get_operators() {
            let tokens = operator_tokens(operator.name());
            if tokens.is_empty() {
                continue;
            }
            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !finite_cost(cost) {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        direct[from][to] = direct[from][to].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_index.get(tokens[1]), place_index.get(tokens[2]))
                    {
                        via_depot[from][to] = via_depot[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = place_index.get(tokens[1]) {
                        direct[from][0] = direct[from][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Phase B: fill only missing direct arcs from exact travel-cost PNEs.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            if id >= initial_numeric.len() {
                continue;
            }
            if let Some((from_name, to_name)) = parse_travel_cost_name(variable.name()) {
                if let (Some(&from), Some(&to)) =
                    (place_index.get(&from_name), place_index.get(&to_name))
                {
                    let value = initial_numeric[id];
                    if !finite_cost(direct[from][to]) && finite_cost(value) {
                        direct[from][to] = value;
                    }
                }
            }
        }

        // Phase C: a via-depot transition is exactly from->depot + depot->to.
        // This only fills entries that grounded via-depot operators did not supply.
        for from in 1..place_count {
            for to in 1..place_count {
                if from == to || finite_cost(via_depot[from][to]) {
                    continue;
                }
                if finite_cost(direct[from][0]) && finite_cost(direct[0][to]) {
                    via_depot[from][to] = direct[from][0] + direct[0][to];
                }
            }
        }

        // Every customer must at least be reachable from and returnable to the
        // depot. Otherwise this matrix cannot support a meaningful estimate.
        for customer_place in 1..place_count {
            if !finite_cost(direct[0][customer_place])
                || !finite_cost(direct[customer_place][0])
            {
                return Err(format!(
                    "llm_001 has an unusable cost matrix for place {}: missing depot arc",
                    place_names[customer_place]
                )
                .into());
            }
        }

        // Precompute directed local connection estimates. Self arcs are ignored.
        let mut min_out = vec![0.0; place_count];
        let mut min_via_from = vec![INF; place_count];
        let mut scale_sum = 0.0;
        let mut scale_count = 0usize;

        for from in 1..place_count {
            let mut best = direct[from][0];
            for to in 1..place_count {
                if from != to && finite_cost(direct[from][to]) {
                    best = best.min(direct[from][to]);
                }
                if from != to && finite_cost(via_depot[from][to]) {
                    min_via_from[from] = min_via_from[from].min(via_depot[from][to]);
                }
            }
            min_out[from] = best;
            if finite_cost(best) {
                scale_sum += best;
                scale_count += 1;
            }
        }

        let arc_scale = if scale_count > 0 {
            scale_sum / scale_count as f64
        } else {
            1.0
        };

        Ok(Self {
            loc_facts,
            customers,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            capacity_known,
            max_vehicles_known,
            direct,
            via_depot,
            min_out,
            min_via_from,
            arc_scale,
            scratch: RefCell::new(Scratch {
                numeric: vec![0.0; task.numeric_variables().len()],
                unvisited: Vec::with_capacity(customer_names.len()),
            }),
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
        let mut scratch = self.scratch.borrow_mut();

        view.fill_numeric(&mut scratch.numeric).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let load = scratch.numeric[self.load_id].max(0.0);
        let vehicles_used = scratch.numeric[self.vehicles_id].max(0.0);

        let current_place = self
            .loc_facts
            .iter()
            .position(|fact| fact.is_hold(view));
        let current_place = match current_place {
            Some(place) => place,
            None => return Ok(LARGE_PENALTY),
        };

        scratch.unvisited.clear();
        let mut remaining_demand = 0.0;
        for (customer_index, customer) in self.customers.iter().enumerate() {
            if !customer.visited_fact.is_hold(view) {
                scratch.unvisited.push(customer_index);
                remaining_demand += customer.demand;
            }
        }

        // All customers are served: the only remaining routing action is the
        // required final return to the depot.
        if scratch.unvisited.is_empty() {
            if current_place == 0 {
                return Ok(0.0);
            }
            let return_cost = self.direct[current_place][0];
            return Ok(if finite_cost(return_cost) {
                return_cost
            } else {
                LARGE_PENALTY
            });
        }

        // Best-effort resource dead-end detection. Missing folded constants and
        // demands were deliberately represented by permissive defaults.
        let mut required_new_vehicles = 0.0;
        if self.capacity_known {
            for &customer_index in &scratch.unvisited {
                if self.customers[customer_index].demand > self.capacity + 1.0e-9 {
                    return Ok(LARGE_PENALTY);
                }
            }

            let residual = (self.capacity - load).max(0.0);
            if remaining_demand > residual {
                required_new_vehicles =
                    ((remaining_demand - residual) / self.capacity).ceil();
            }

            if self.max_vehicles_known {
                let remaining_fleet = (self.max_vehicles - vehicles_used).max(0.0);
                let available_capacity = residual + remaining_fleet * self.capacity;
                if remaining_demand > available_capacity + 1.0e-9
                    || required_new_vehicles > remaining_fleet + 1.0e-9
                {
                    return Ok(LARGE_PENALTY);
                }
            }
        }

        // Select a state-anchored first transition. Direct visit is preferred
        // while at least one unvisited demand fits the current load. If none
        // fits, use the domain's visit-via-depot transition, subject to fleet
        // availability. This is recomputed for every state.
        let mut first_cost = INF;
        let mut first_customer = None;
        let mut first_was_via = false;

        for &customer_index in &scratch.unvisited {
            let place = customer_index + 1;
            let demand = self.customers[customer_index].demand;
            let fits = !self.capacity_known
                || current_place == 0
                || load + demand <= self.capacity + 1.0e-9;
            let cost = self.direct[current_place][place];
            if fits && finite_cost(cost) && cost < first_cost {
                first_cost = cost;
                first_customer = Some(customer_index);
            }
        }

        if first_customer.is_none() && current_place != 0 {
            let fleet_available = !self.max_vehicles_known
                || vehicles_used + 1.0 <= self.max_vehicles + 1.0e-9;
            if fleet_available {
                for &customer_index in &scratch.unvisited {
                    let place = customer_index + 1;
                    let cost = self.via_depot[current_place][place];
                    if finite_cost(cost) && cost < first_cost {
                        first_cost = cost;
                        first_customer = Some(customer_index);
                        first_was_via = true;
                    }
                }
            }
        }

        let first_customer = match first_customer {
            Some(customer) => customer,
            None => return Ok(LARGE_PENALTY),
        };

        // h_routing: current-location anchor, one cheap directed continuation
        // per remaining customer except the selected first customer, and an
        // explicit final return-to-depot edge.
        let mut h_routing = first_cost;
        let mut final_return = INF;
        for &customer_index in &scratch.unvisited {
            let place = customer_index + 1;
            if customer_index != first_customer {
                h_routing += self.min_out[place];
            }
            final_return = final_return.min(self.direct[place][0]);
        }
        if !finite_cost(final_return) {
            return Ok(LARGE_PENALTY);
        }
        h_routing += final_return;

        // h_capacity: estimate the extra travel introduced by future depot
        // reloads. A reload already represented by the first via-depot move is
        // not counted twice.
        let future_reloads =
            (required_new_vehicles - if first_was_via { 1.0 } else { 0.0 }).max(0.0);
        let mut cheapest_break_extra = INF;
        for &customer_index in &scratch.unvisited {
            let place = customer_index + 1;
            if finite_cost(self.min_via_from[place]) {
                cheapest_break_extra = cheapest_break_extra
                    .min((self.min_via_from[place] - self.min_out[place]).max(0.0));
            }
        }
        if !finite_cost(cheapest_break_extra) {
            cheapest_break_extra = self.arc_scale;
        }
        let h_capacity = future_reloads * cheapest_break_extra;

        // h_vehicles is a small fleet-pressure tie breaker. It distinguishes
        // states that require the same number of reloads but have consumed
        // different portions of a small fleet.
        let fleet_pressure = if self.max_vehicles_known && self.max_vehicles > 0.0 {
            (vehicles_used / self.max_vehicles).clamp(0.0, 1.0)
        } else {
            vehicles_used / (vehicles_used + 1.0)
        };
        let h_vehicles = if required_new_vehicles > 0.0 {
            0.1 * self.arc_scale * required_new_vehicles * fleet_pressure
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
