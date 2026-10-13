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

const DEAD_END_PENALTY: f64 = 1.0e15;
const INF: f64 = f64::INFINITY;

fn unary_atom_argument(name: &str, predicate: &str) -> Option<String> {
    let mut text = name.trim();
    if let Some(rest) = text.strip_prefix("Atom ") {
        text = rest.trim();
    }

    let rest = text.strip_prefix(predicate)?;
    let rest = rest.strip_prefix('(')?;
    let argument = rest.strip_suffix(')')?.trim();
    if argument.is_empty() || argument.contains(',') {
        return None;
    }
    Some(argument.to_string())
}

fn clean_operator_token(token: &str) -> &str {
    token.trim_matches(|c| c == '(' || c == ')')
}

fn exact_numeric_id(
    task: &dyn AbstractNumericTask,
    target: &str,
) -> Option<usize> {
    for (id, variable) in task.numeric_variables().iter().enumerate() {
        let name = variable.name().trim();
        if name == target {
            return Some(id);
        }
        if let Some(unwrapped) = name.strip_prefix("PNE ") {
            if unwrapped.trim() == target {
                return Some(id);
            }
        }
    }
    None
}

fn exact_numeric_initial_value(
    task: &dyn AbstractNumericTask,
    target: &str,
) -> Option<f64> {
    let id = exact_numeric_id(task, target)?;
    task.get_initial_numeric_state_values().get(id).copied()
}

struct LlmHeuristic {
    // Node zero is the structurally discovered depot; all other nodes are
    // customers. Matrices remain directed.
    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,

    loc_facts: Vec<ExplicitFact>,
    visited_facts: Vec<Option<ExplicitFact>>,

    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_id: usize,
    vehicles_id: usize,

    // A spatial backbone built once from directed visit costs. Evaluation
    // rotates and filters it according to the actual state.
    backbone: Vec<usize>,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover all semantic loc and visited facts. We deliberately do not
        // infer semantics from generic SAS variable names such as var0.
        let mut loc_entries: Vec<(String, ExplicitFact)> = Vec::new();
        let mut visited_entries: Vec<(String, ExplicitFact)> = Vec::new();
        let mut seen_loc: HashSet<String> = HashSet::new();
        let mut seen_visited: HashSet<String> = HashSet::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = unary_atom_argument(fact_name, "loc") {
                    if seen_loc.insert(object.clone()) {
                        loc_entries.push((object, fact));
                    }
                }
                if let Some(object) = unary_atom_argument(fact_name, "visited") {
                    if seen_visited.insert(object.clone()) {
                        visited_entries.push((object, fact));
                    }
                }
            }
        }

        if loc_entries.is_empty() || visited_entries.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not discover CVRP loc/visited facts from semantic fact names"
                    .to_string(),
            ));
        }

        let customer_set: HashSet<String> = visited_entries
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        let depot_candidates: Vec<String> = loc_entries
            .iter()
            .filter(|(name, _)| !customer_set.contains(name))
            .map(|(name, _)| name.clone())
            .collect();

        if depot_candidates.len() != 1 {
            return Err(HeuristicBuildError::from(format!(
                "llm_001 expected one locatable non-customer depot, found {}",
                depot_candidates.len()
            )));
        }
        let depot_name = depot_candidates[0].clone();

        // Stable node ordering: depot first, then customers in semantic-fact
        // scan order. No assumptions are made about customer spelling.
        let mut node_names: Vec<String> = Vec::with_capacity(visited_entries.len() + 1);
        node_names.push(depot_name.clone());
        for (name, _) in &visited_entries {
            node_names.push(name.clone());
        }

        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let mut loc_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        for (name, fact) in loc_entries {
            loc_by_name.insert(name, fact);
        }
        let mut visited_by_name: HashMap<String, ExplicitFact> = HashMap::new();
        for (name, fact) in visited_entries {
            visited_by_name.insert(name, fact);
        }

        let mut loc_facts: Vec<ExplicitFact> = Vec::with_capacity(node_names.len());
        let mut visited_facts: Vec<Option<ExplicitFact>> =
            Vec::with_capacity(node_names.len());

        for (index, name) in node_names.iter().enumerate() {
            let loc_fact = loc_by_name.remove(name).ok_or_else(|| -> HeuristicBuildError {
                format!("llm_001 found no loc fact for routing object {name}").into()
            })?;
            loc_facts.push(loc_fact);

            if index == 0 {
                visited_facts.push(None);
            } else {
                let visited_fact =
                    visited_by_name
                        .remove(name)
                        .ok_or_else(|| -> HeuristicBuildError {
                            format!("llm_001 found no visited fact for customer {name}").into()
                        })?;
                visited_facts.push(Some(visited_fact));
            }
        }

        let load_id = exact_numeric_id(task, "load()").ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 could not find the required exact numeric fluent load()"
                    .to_string()
                    .into()
            },
        )?;
        let vehicles_id = exact_numeric_id(task, "vehicles()").ok_or_else(
            || -> HeuristicBuildError {
                "llm_001 could not find the required exact numeric fluent vehicles()"
                    .to_string()
                    .into()
            },
        )?;

        // These read-only fluents may have been folded away by translation.
        // Exact lookup only; permissive defaults preserve the routing term.
        let capacity = exact_numeric_initial_value(task, "capacity()")
            .filter(|x| x.is_finite() && *x >= 0.0)
            .unwrap_or(f64::MAX);
        let max_vehicles = exact_numeric_initial_value(task, "max_vehicles()")
            .filter(|x| x.is_finite() && *x >= 0.0)
            .unwrap_or(f64::MAX);

        let mut demands: Vec<f64> = vec![0.0; node_names.len()];
        for customer in 1..node_names.len() {
            let exact_name = format!("demand({})", node_names[customer]);
            demands[customer] = exact_numeric_initial_value(task, &exact_name)
                .filter(|x| x.is_finite() && *x >= 0.0)
                .unwrap_or(0.0);
        }

        let n = node_names.len();
        let mut direct_cost: Vec<Vec<f64>> = vec![vec![INF; n]; n];
        let mut via_depot_cost: Vec<Vec<f64>> = vec![vec![INF; n]; n];
        let mut return_cost: Vec<f64> = vec![INF; n];
        direct_cost[0][0] = 0.0;
        return_cost[0] = 0.0;

        // Phase A: primary reconstruction from grounded operators and their
        // metric costs. This avoids reliance on folded travel-cost PNE names.
        for operator in task.get_operators() {
            let tokens: Vec<&str> = operator
                .name()
                .split_whitespace()
                .map(clean_operator_token)
                .filter(|token| !token.is_empty())
                .collect();
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        direct_cost[from][to] = direct_cost[from][to].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        via_depot_cost[from][to] =
                            via_depot_cost[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = node_index.get(tokens[1]) {
                        return_cost[from] = return_cost[from].min(cost);
                        direct_cost[from][0] = direct_cost[from][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Phase C: fill only missing arcs via known depot legs. For this CVRP
        // domain this is a real executable route boundary, not a symmetry
        // assumption. Grounded visit-via-depot costs remain preferred.
        for from in 1..n {
            for to in 1..n {
                if from == to {
                    direct_cost[from][to] = 0.0;
                    via_depot_cost[from][to] = 0.0;
                    continue;
                }
                if !via_depot_cost[from][to].is_finite()
                    && return_cost[from].is_finite()
                    && direct_cost[0][to].is_finite()
                {
                    via_depot_cost[from][to] =
                        return_cost[from] + direct_cost[0][to];
                }
                if !direct_cost[from][to].is_finite()
                    && return_cost[from].is_finite()
                    && direct_cost[0][to].is_finite()
                {
                    // Conservative usable completion for a missing direct arc.
                    direct_cost[from][to] =
                        return_cost[from] + direct_cost[0][to];
                }
            }
        }

        // The depot departure, final return, and all customer transitions must
        // be usable after conservative completion.
        for customer in 1..n {
            if !direct_cost[0][customer].is_finite()
                || !return_cost[customer].is_finite()
            {
                return Err(HeuristicBuildError::from(format!(
                    "llm_001 could not reconstruct depot routing costs for {}",
                    node_names[customer]
                )));
            }
            for other in 1..n {
                if customer != other
                    && (!direct_cost[customer][other].is_finite()
                        || !via_depot_cost[customer][other].is_finite())
                {
                    return Err(HeuristicBuildError::from(format!(
                        "llm_001 routing matrix remains unusable between {} and {}",
                        node_names[customer], node_names[other]
                    )));
                }
            }
        }

        // Precompute a directed nearest-neighbor spatial backbone once. It is
        // not blindly replayed: each evaluation filters visited nodes, rotates
        // to the best transition from the current location, and inserts route
        // boundaries according to current load and fleet usage.
        let mut backbone: Vec<usize> = Vec::with_capacity(n.saturating_sub(1));
        let mut selected = vec![false; n];
        selected[0] = true;
        let mut current = 0usize;
        for _ in 1..n {
            let mut best: Option<usize> = None;
            let mut best_cost: f64 = INF;
            for candidate in 1..n {
                if !selected[candidate] && direct_cost[current][candidate] < best_cost {
                    best = Some(candidate);
                    best_cost = direct_cost[current][candidate];
                }
            }
            let next = best.ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not construct its routing backbone"
                    .to_string()
                    .into()
            })?;
            selected[next] = true;
            backbone.push(next);
            current = next;
        }

        Ok(Self {
            direct_cost,
            via_depot_cost,
            return_cost,
            loc_facts,
            visited_facts,
            demands,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
            backbone,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; n]),
        })
    }

    fn fits(&self, load: f64, customer: usize) -> bool {
        if self.capacity == f64::MAX {
            return true;
        }
        load + self.demands[customer] <= self.capacity + 1.0e-9
    }

    fn has_another_vehicle(&self, vehicles: f64) -> bool {
        if self.max_vehicles == f64::MAX {
            return true;
        }
        vehicles + 1.0 <= self.max_vehicles + 1.0e-9
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
            .fill_numeric(&mut numeric_values)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!(
                    "llm_001 could not read numeric state: {err:?}"
                ))
            })?;

        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        let mut current_location: Option<usize> = None;
        for (node, fact) in self.loc_facts.iter().enumerate() {
            if propositional_values[fact.variable()] == fact.value() {
                current_location = Some(node);
                break;
            }
        }
        let mut current = current_location.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "llm_001 found no active loc fact in the evaluated state".to_string(),
            )
        })?;

        let current_load = numeric_values[self.load_id].max(0.0);
        let current_vehicles = numeric_values[self.vehicles_id].max(0.0);

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        unvisited.fill(false);
        let mut remaining_count = 0usize;
        let mut remaining_demand: f64 = 0.0;
        let mut minimum_round_trip: f64 = INF;

        for customer in 1..self.loc_facts.len() {
            let fact = self.visited_facts[customer]
                .as_ref()
                .expect("customer node must have a visited fact");
            let is_visited = propositional_values[fact.variable()] == fact.value();
            if !is_visited {
                unvisited[customer] = true;
                remaining_count += 1;
                remaining_demand += self.demands[customer];
                minimum_round_trip = minimum_round_trip.min(
                    self.direct_cost[0][customer] + self.return_cost[customer],
                );
            }
        }

        // All service goals are achieved, but the explicit location goal may
        // still require the domain's return-to-depot action.
        if remaining_count == 0 {
            if current == 0 {
                return Ok(0.0);
            }
            return Ok(self.return_cost[current].min(DEAD_END_PENALTY));
        }

        // Best-effort resource dead-end detection. If read-only PNEs were
        // folded away, defaults make this check permissive rather than
        // suppressing the routing estimate.
        if self.capacity != f64::MAX && self.max_vehicles != f64::MAX {
            let residual_capacity = (self.capacity - current_load).max(0.0);
            let remaining_vehicle_slots =
                (self.max_vehicles - current_vehicles).max(0.0).floor();
            let total_available =
                residual_capacity + remaining_vehicle_slots * self.capacity;

            let oversized_customer = (1..self.loc_facts.len()).any(|customer| {
                unvisited[customer]
                    && self.demands[customer] > self.capacity + 1.0e-9
            });

            if oversized_customer
                || remaining_demand > total_available + 1.0e-9
                || current_vehicles > self.max_vehicles + 1.0e-9
            {
                return Ok(DEAD_END_PENALTY);
            }
        }

        // Re-anchor the precomputed spatial backbone at the best currently
        // executable first customer. This first transition explicitly differs
        // between a direct visit and a visit-via-depot reload.
        let mut anchor_position: Option<usize> = None;
        let mut anchor_cost: f64 = INF;
        for (position, &customer) in self.backbone.iter().enumerate() {
            if !unvisited[customer] {
                continue;
            }

            let transition = if current == 0 {
                if self.fits(current_load, customer) {
                    self.direct_cost[0][customer]
                } else {
                    INF
                }
            } else if self.fits(current_load, customer) {
                self.direct_cost[current][customer]
            } else if self.has_another_vehicle(current_vehicles) {
                self.via_depot_cost[current][customer]
            } else {
                INF
            };

            if transition < anchor_cost {
                anchor_cost = transition;
                anchor_position = Some(position);
            }
        }

        let anchor = match anchor_position {
            Some(position) => position,
            None => return Ok(DEAD_END_PENALTY),
        };

        // h_routing: filtered, state-anchored capacitated completion. The load
        // and vehicle counters evolve exactly according to the two routing
        // action schemas, followed by the mandatory final depot return.
        let mut h_routing: f64 = 0.0;
        let mut simulated_load = current_load;
        let mut simulated_vehicles = current_vehicles;
        let mut served = 0usize;

        for offset in 0..self.backbone.len() {
            let position = (anchor + offset) % self.backbone.len();
            let customer = self.backbone[position];
            if !unvisited[customer] {
                continue;
            }

            if current == 0 {
                if !self.fits(simulated_load, customer) {
                    return Ok(DEAD_END_PENALTY);
                }
                h_routing += self.direct_cost[0][customer];
                simulated_load += self.demands[customer];
            } else if self.fits(simulated_load, customer) {
                h_routing += self.direct_cost[current][customer];
                simulated_load += self.demands[customer];
            } else if self.has_another_vehicle(simulated_vehicles) {
                h_routing += self.via_depot_cost[current][customer];
                simulated_vehicles += 1.0;
                simulated_load = self.demands[customer];
            } else {
                return Ok(DEAD_END_PENALTY);
            }

            current = customer;
            served += 1;
        }

        if served != remaining_count || current == 0 {
            return Ok(DEAD_END_PENALTY);
        }
        h_routing += self.return_cost[current];

        // h_resource: a small fleet/capacity pressure component. It estimates
        // the minimum number of additional route starts implied by aggregate
        // demand after using the residual load capacity. Pressure rises as
        // those required starts consume the remaining fleet slots.
        let mut h_resource: f64 = 0.0;
        if self.capacity != f64::MAX && self.capacity > 0.0 {
            let residual = (self.capacity - current_load).max(0.0);
            let demand_after_current_route = (remaining_demand - residual).max(0.0);
            let required_reloads =
                (demand_after_current_route / self.capacity).ceil();

            if required_reloads > 0.0 && minimum_round_trip.is_finite() {
                let mut fleet_pressure: f64 = 0.0;
                if self.max_vehicles != f64::MAX {
                    let slots = (self.max_vehicles - current_vehicles).max(0.0);
                    if slots <= 0.0 {
                        return Ok(DEAD_END_PENALTY);
                    }
                    fleet_pressure = (required_reloads / slots).min(1.0);
                }
                h_resource = required_reloads
                    * minimum_round_trip
                    * (0.15 + 0.10 * fleet_pressure);
            }
        }

        let h = h_routing + h_resource;
        if !h.is_finite() {
            return Ok(DEAD_END_PENALTY);
        }
        Ok(h.max(0.0).min(DEAD_END_PENALTY))
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
    planforge::run_with_heuristics(vec![ExternalHeuristic {
        name: "llm_001",
        backend: RequiredBackend::None,
        requirements: any_task,
        nested_heuristics: no_nested_heuristics,
        build: build_llm_001,
    }])
}
