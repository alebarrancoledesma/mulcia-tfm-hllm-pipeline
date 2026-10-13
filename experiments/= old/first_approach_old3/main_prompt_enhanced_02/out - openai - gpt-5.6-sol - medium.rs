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

const DEAD_END_PENALTY: f64 = 1.0e15;

fn build_error(message: &str) -> HeuristicBuildError {
    message.to_string().into()
}

fn atom_argument(fact_name: &str, predicate: &str) -> Option<String> {
    let needle = format!("{}(", predicate);
    let start = fact_name.find(&needle)? + needle.len();
    let suffix = &fact_name[start..];
    let end = suffix.find(')')?;
    let argument = suffix[..end].trim();
    if argument.is_empty() {
        None
    } else {
        Some(argument.to_string())
    }
}

fn unary_numeric_argument(name: &str, function: &str) -> Option<String> {
    let needle = format!("{}(", function);
    let start = name.find(&needle)? + needle.len();
    let suffix = &name[start..];
    let end = suffix.find(')')?;
    let argument = suffix[..end].trim();
    if argument.is_empty() {
        None
    } else {
        Some(argument.to_string())
    }
}

fn binary_numeric_arguments(name: &str, function: &str) -> Option<(String, String)> {
    let needle = format!("{}(", function);
    let start = name.find(&needle)? + needle.len();
    let suffix = &name[start..];
    let end = suffix.find(')')?;
    let inside = suffix[..end].replace(',', " ");
    let mut parts = inside.split_whitespace();
    let from = parts.next()?.to_string();
    let to = parts.next()?.to_string();
    if parts.next().is_some() {
        None
    } else {
        Some((from, to))
    }
}

fn is_exact_zero_ary_pne(name: &str, function: &str) -> bool {
    let trimmed = name.trim();
    trimmed == format!("PNE {}()", function) || trimmed == format!("{}()", function)
}

fn has_zero_ary_suffix(name: &str, function: &str) -> bool {
    name.trim().ends_with(&format!("{}()", function))
}

struct LlmHeuristic {
    node_count: usize,
    depot: usize,
    customer_nodes: Vec<usize>,
    loc_facts: Vec<ExplicitFact>,
    visited_facts: Vec<ExplicitFact>,
    demands: Vec<f64>,
    direct_cost: Vec<f64>,
    via_depot_cost: Vec<f64>,
    nearest_from: Vec<Vec<usize>>,
    tour_order: Vec<usize>,
    tour_rank: Vec<usize>,
    capacity: f64,
    max_vehicles: f64,
    load_id: usize,
    vehicles_id: usize,
    route_overhead: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propositional: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn matrix_index(&self, from: usize, to: usize) -> usize {
        from * self.node_count + to
    }

    fn direct(&self, from: usize, to: usize) -> f64 {
        self.direct_cost[self.matrix_index(from, to)]
    }

    fn via_depot(&self, from: usize, to: usize) -> f64 {
        self.via_depot_cost[self.matrix_index(from, to)]
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // Discover semantic propositional facts. SAS variable names are not used.
        let mut loc_by_object: HashMap<String, ExplicitFact> = HashMap::new();
        let mut visited_by_object: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                // Positive semantic facts are named "Atom ..." in the observed
                // compilation. This avoids accidentally treating a negated fact
                // representation as the positive visited/loc fact.
                if !fact_name.contains("Atom ") {
                    continue;
                }

                if let Some(object) = atom_argument(&fact_name, "loc") {
                    loc_by_object.insert(object, fact.clone());
                }
                if let Some(object) = atom_argument(&fact_name, "visited") {
                    visited_by_object.insert(object, fact.clone());
                }
            }
        }

        if visited_by_object.is_empty() {
            return Err(build_error("llm_001: no semantic visited(customer) facts found"));
        }

        // The unique location object without a visited predicate is the depot.
        let mut depot_objects: Vec<String> = loc_by_object
            .keys()
            .filter(|name| !visited_by_object.contains_key(*name))
            .cloned()
            .collect();
        depot_objects.sort();

        if depot_objects.len() != 1 {
            return Err(build_error(
                "llm_001: expected one depot identifiable from loc facts without visited facts",
            ));
        }
        let depot_name = depot_objects[0].clone();

        let mut customer_names: Vec<String> = visited_by_object.keys().cloned().collect();
        customer_names.sort();

        for customer in &customer_names {
            if !loc_by_object.contains_key(customer) {
                return Err(build_error(
                    "llm_001: a visited customer has no corresponding loc fact",
                ));
            }
        }

        // Node zero is structural, not tied to any PDDL object spelling.
        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name.clone());
        node_names.extend(customer_names.iter().cloned());
        let node_count = node_names.len();
        let depot = 0usize;

        let mut node_by_name = HashMap::new();
        for (node, name) in node_names.iter().enumerate() {
            node_by_name.insert(name.clone(), node);
        }

        let mut loc_facts = Vec::with_capacity(node_count);
        for name in &node_names {
            loc_facts.push(
                loc_by_object
                    .get(name)
                    .ok_or_else(|| build_error("llm_001: incomplete loc fact mapping"))?
                    .clone(),
            );
        }

        let mut visited_facts = Vec::with_capacity(customer_names.len());
        for name in &customer_names {
            visited_facts.push(
                visited_by_object
                    .get(name)
                    .ok_or_else(|| build_error("llm_001: incomplete visited fact mapping"))?
                    .clone(),
            );
        }

        let customer_nodes: Vec<usize> = (1..node_count).collect();
        let initial_numeric = task.get_initial_numeric_state_values();

        // Resolve frequently-read dynamic numeric IDs exactly once.
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut demands = vec![f64::NAN; node_count];
        demands[depot] = 0.0;

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if is_exact_zero_ary_pne(name, "load") {
                load_id = Some(id);
            } else if is_exact_zero_ary_pne(name, "vehicles") {
                vehicles_id = Some(id);
            }

            if has_zero_ary_suffix(name, "capacity") {
                capacity = initial_numeric.get(id).copied();
            }
            if has_zero_ary_suffix(name, "max_vehicles") {
                max_vehicles = initial_numeric.get(id).copied();
            }
            if let Some(customer) = unary_numeric_argument(name, "demand") {
                if let (Some(&node), Some(&value)) =
                    (node_by_name.get(&customer), initial_numeric.get(id))
                {
                    if node != depot {
                        demands[node] = value;
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            build_error("llm_001: dynamic numeric fluent load() was not found")
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            build_error("llm_001: dynamic numeric fluent vehicles() was not found")
        })?;
        let capacity = capacity.ok_or_else(|| {
            build_error("llm_001: capacity could not be recovered from numeric task data")
        })?;
        let max_vehicles = max_vehicles.ok_or_else(|| {
            build_error("llm_001: max_vehicles could not be recovered from numeric task data")
        })?;

        if !capacity.is_finite() || capacity <= 0.0 {
            return Err(build_error("llm_001: capacity is not positive and finite"));
        }
        if !max_vehicles.is_finite() || max_vehicles < 1.0 {
            return Err(build_error("llm_001: max_vehicles is not positive and finite"));
        }
        if customer_nodes
            .iter()
            .any(|&node| !demands[node].is_finite() || demands[node] < 0.0)
        {
            return Err(build_error(
                "llm_001: one or more customer demands could not be recovered",
            ));
        }

        let matrix_len = node_count * node_count;
        let mut direct_cost = vec![f64::INFINITY; matrix_len];
        let mut via_depot_cost = vec![f64::INFINITY; matrix_len];
        for node in 0..node_count {
            direct_cost[node * node_count + node] = 0.0;
            via_depot_cost[node * node_count + node] = 0.0;
        }

        // Phase A: primary extraction from grounded operator semantics.
        let mut operator_arcs = 0usize;
        for operator in task.get_operators() {
            let words: Vec<&str> = operator.name().split_whitespace().collect();
            if words.is_empty() {
                continue;
            }
            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match words.as_slice() {
                ["visit", from, to] => {
                    if let (Some(&from_node), Some(&to_node)) =
                        (node_by_name.get(*from), node_by_name.get(*to))
                    {
                        direct_cost[from_node * node_count + to_node] = cost;
                        operator_arcs += 1;
                    }
                }
                ["visit-via-depot", from, to] => {
                    if let (Some(&from_node), Some(&to_node)) =
                        (node_by_name.get(*from), node_by_name.get(*to))
                    {
                        via_depot_cost[from_node * node_count + to_node] = cost;
                        operator_arcs += 1;
                    }
                }
                ["return-to-depot", from] => {
                    if let Some(&from_node) = node_by_name.get(*from) {
                        direct_cost[from_node * node_count + depot] = cost;
                        operator_arcs += 1;
                    }
                }
                _ => {}
            }
        }

        if operator_arcs == 0 {
            return Err(build_error(
                "llm_001: no routing costs could be extracted from grounded operators",
            ));
        }

        // Phase B: fill missing direct arcs from explicit travel-cost PNEs when
        // such names survive compilation. This never overwrites operator data.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if name.contains("travel-cost-via-depot") {
                continue;
            }
            if let Some((from, to)) = binary_numeric_arguments(name, "travel-cost") {
                if let (Some(&from_node), Some(&to_node), Some(&value)) = (
                    node_by_name.get(&from),
                    node_by_name.get(&to),
                    initial_numeric.get(id),
                ) {
                    let index = from_node * node_count + to_node;
                    if !direct_cost[index].is_finite() && value.is_finite() && value >= 0.0 {
                        direct_cost[index] = value;
                    }
                }
            }
        }

        // Phase C: conservatively complete only missing customer arcs through
        // the depot. Direction is retained; symmetry is never assumed.
        for from in 0..node_count {
            for to in 0..node_count {
                if from == to {
                    continue;
                }
                let index = from * node_count + to;
                if !direct_cost[index].is_finite() {
                    let back = direct_cost[from * node_count + depot];
                    let out = direct_cost[depot * node_count + to];
                    if back.is_finite() && out.is_finite() {
                        direct_cost[index] = back + out;
                    }
                }
                if !via_depot_cost[index].is_finite() {
                    let back = direct_cost[from * node_count + depot];
                    let out = direct_cost[depot * node_count + to];
                    if back.is_finite() && out.is_finite() {
                        via_depot_cost[index] = back + out;
                    }
                }
            }
        }

        if direct_cost.iter().any(|value| !value.is_finite()) {
            return Err(build_error(
                "llm_001: routing matrix remains incomplete after operator, PNE, and depot-path extraction",
            ));
        }

        for &from in &customer_nodes {
            for &to in &customer_nodes {
                if from != to && !via_depot_cost[from * node_count + to].is_finite() {
                    return Err(build_error(
                        "llm_001: visit-via-depot costs remain incomplete",
                    ));
                }
            }
        }

        // Sorted adjacency is constructed once. Evaluation only filters these
        // lists against the state's unvisited flags.
        let mut nearest_from = Vec::with_capacity(node_count);
        for from in 0..node_count {
            let mut neighbors: Vec<usize> = customer_nodes.clone();
            neighbors.sort_by(|&left, &right| {
                direct_cost[from * node_count + left]
                    .total_cmp(&direct_cost[from * node_count + right])
                    .then_with(|| left.cmp(&right))
            });
            nearest_from.push(neighbors);
        }

        // A task-level directed nearest-neighbor cycle supplies geographic
        // coherence, but each evaluation re-anchors and filters it using the
        // actual current location and unvisited set.
        let mut tour_order = Vec::with_capacity(customer_nodes.len());
        let mut used = vec![false; node_count];
        let mut cursor = depot;
        while tour_order.len() < customer_nodes.len() {
            let next = nearest_from[cursor]
                .iter()
                .copied()
                .find(|&node| !used[node])
                .ok_or_else(|| build_error("llm_001: failed to construct customer ordering"))?;
            used[next] = true;
            tour_order.push(next);
            cursor = next;
        }

        let mut tour_rank = vec![0usize; node_count];
        for (rank, &node) in tour_order.iter().enumerate() {
            tour_rank[node] = rank;
        }

        let min_out = customer_nodes
            .iter()
            .map(|&node| direct_cost[depot * node_count + node])
            .fold(f64::INFINITY, f64::min);
        let min_back = customer_nodes
            .iter()
            .map(|&node| direct_cost[node * node_count + depot])
            .fold(f64::INFINITY, f64::min);
        let route_overhead = if min_out.is_finite() && min_back.is_finite() {
            min_out + min_back
        } else {
            0.0
        };

        Ok(Self {
            node_count,
            depot,
            customer_nodes,
            loc_facts,
            visited_facts,
            demands,
            direct_cost,
            via_depot_cost,
            nearest_from,
            tour_order,
            tour_rank,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
            route_overhead,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propositional: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; node_count]),
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

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        view.fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let mut propositional_values = self.scratch_propositional.borrow_mut();
        state.fill_state(registry, &mut propositional_values);

        let current = self
            .loc_facts
            .iter()
            .position(|fact| fact.is_hold(view))
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "llm_001: state has no recognized current location".to_string(),
                )
            })?;

        let current_load = numeric_values
            .get(self.load_id)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let vehicles_used = numeric_values
            .get(self.vehicles_id)
            .copied()
            .unwrap_or(self.max_vehicles)
            .max(0.0);

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        unvisited.fill(false);

        let mut remaining_demand = 0.0f64;
        let mut remaining_count = 0usize;
        for (customer_index, &node) in self.customer_nodes.iter().enumerate() {
            if !self.visited_facts[customer_index].is_hold(view) {
                unvisited[node] = true;
                remaining_demand += self.demands[node];
                remaining_count += 1;
            }
        }

        // Required edge case: all customers served but final depot return is
        // still outstanding.
        if remaining_count == 0 {
            if current == self.depot {
                return Ok(0.0);
            }
            return Ok(self.direct(current, self.depot).max(0.0));
        }

        let residual_capacity = (self.capacity - current_load).max(0.0);
        let remaining_vehicle_slots = (self.max_vehicles - vehicles_used).max(0.0);
        let total_available_capacity =
            residual_capacity + remaining_vehicle_slots * self.capacity;

        // Necessary aggregate fleet-capacity condition.
        if remaining_demand > total_available_capacity + 1.0e-9 {
            return Ok(DEAD_END_PENALTY + remaining_demand);
        }

        // Detect states with no legal next service action. From a customer a
        // via-depot action is possible whenever a fleet slot remains; from the
        // depot only a capacity-feasible direct visit can start a route.
        let has_directly_fitting_customer = self
            .customer_nodes
            .iter()
            .copied()
            .any(|node| unvisited[node] && self.demands[node] <= residual_capacity + 1.0e-9);
        if !has_directly_fitting_customer
            && (current == self.depot || remaining_vehicle_slots < 1.0 - 1.0e-9)
        {
            return Ok(DEAD_END_PENALTY + remaining_demand);
        }

        // Re-anchor the static geographic cycle at the unvisited customer that
        // is nearest to the actual current state location.
        let anchor = self.nearest_from[current]
            .iter()
            .copied()
            .find(|&node| unvisited[node])
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "llm_001: unvisited set has no routing anchor".to_string(),
                )
            })?;
        let anchor_rank = self.tour_rank[anchor];

        // Component 1: state-dependent capacitated routing completion. Direct
        // visit and visit-via-depot use separately reconstructed action costs.
        let mut h_routing = 0.0f64;
        let mut simulated_location = current;
        let mut simulated_load = current_load;
        let mut simulated_vehicles = vehicles_used;
        let mut simulated_served = 0usize;

        for offset in 0..self.tour_order.len() {
            let rank = (anchor_rank + offset) % self.tour_order.len();
            let next = self.tour_order[rank];
            if !unvisited[next] {
                continue;
            }

            let demand = self.demands[next];
            if simulated_load + demand <= self.capacity + 1.0e-9 {
                h_routing += self.direct(simulated_location, next);
                simulated_load += demand;
            } else if simulated_location != self.depot
                && simulated_vehicles + 1.0 <= self.max_vehicles + 1.0e-9
            {
                h_routing += self.via_depot(simulated_location, next);
                simulated_vehicles += 1.0;
                simulated_load = demand;
            } else {
                // The selected completion exhausted the fleet. This can be a
                // genuine dead end or a poor packing order, so use a large but
                // finite fleet-pressure value rather than panic or infinity.
                return Ok(DEAD_END_PENALTY * 0.5 + h_routing + remaining_demand);
            }

            simulated_location = next;
            simulated_served += 1;
        }

        if simulated_served != remaining_count {
            return Ok(DEAD_END_PENALTY * 0.5 + remaining_demand);
        }

        // The goal requires the vehicle at the depot.
        h_routing += self.direct(simulated_location, self.depot);

        // Component 2: unavoidable reload/fleet-pressure cost. It uses total
        // heterogeneous demand, residual load, capacity, and vehicles already
        // consumed. The completion already models concrete reload locations;
        // this smaller term emphasizes states leaving little fleet slack.
        let demand_after_current_route =
            (remaining_demand - residual_capacity).max(0.0);
        let required_new_vehicles = if demand_after_current_route <= 1.0e-9 {
            0.0
        } else {
            (demand_after_current_route / self.capacity).ceil()
        };
        let fleet_pressure = if required_new_vehicles <= 0.0 {
            0.0
        } else {
            required_new_vehicles / remaining_vehicle_slots.max(1.0)
        };
        let h_capacity = self.route_overhead
            * required_new_vehicles
            * (0.20 + 0.15 * fleet_pressure);

        let h = h_routing.max(0.0) + h_capacity.max(0.0);
        if h.is_finite() {
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
