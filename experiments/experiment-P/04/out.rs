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

const INF: f64 = f64::INFINITY;

struct LlmHeuristic {
    customer_count: usize,
    depot: usize,
    visited_facts: Vec<ExplicitFact>,
    location_facts: Vec<ExplicitFact>,
    direct_cost: Vec<f64>,
    via_depot_cost: Vec<f64>,
    demands: Vec<f64>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    numeric_count: usize,
}

fn atom_argument<'a>(fact_name: &'a str, predicate: &str) -> Option<&'a str> {
    let text = fact_name.trim();
    let text = text.strip_prefix("Atom ").unwrap_or(text).trim();
    let prefix = format!("{}(", predicate);
    let rest = text.strip_prefix(&prefix)?;
    rest.strip_suffix(')').map(str::trim)
}

fn numeric_name_is(name: &str, target: &str) -> bool {
    let name = name.trim();
    name == target || name == format!("PNE {}", target)
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.split_whitespace()
        .map(|token| token.trim_matches(|c| c == '(' || c == ')'))
        .filter(|token| !token.is_empty())
        .collect()
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
            let name = task.get_fact_name(&goal);

            if let Some(customer) = atom_argument(name, "visited") {
                if !customer_names.iter().any(|known| known == customer) {
                    customer_names.push(customer.to_string());
                }
            } else if let Some(place) = atom_argument(name, "loc") {
                depot_name = Some(place.to_string());
            }
        }

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not identify any visited-customer goals".to_string(),
            ));
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not identify the depot location goal"
                .to_string()
                .into()
        })?;

        let customer_count = customer_names.len();
        let depot = customer_count;
        let node_count = customer_count + 1;

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        for (index, name) in customer_names.iter().enumerate() {
            node_by_name.insert(name.clone(), index);
        }
        node_by_name.insert(depot_name.clone(), depot);

        let mut visited_facts: Vec<Option<ExplicitFact>> =
            (0..customer_count).map(|_| None).collect();
        let mut location_facts: Vec<Option<ExplicitFact>> =
            (0..node_count).map(|_| None).collect();

        for variable_id in 0..task.variables().len() {
            let domain_size = task.variables()[variable_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = atom_argument(fact_name, "visited") {
                    if let Some(&index) = node_by_name.get(customer) {
                        if index < customer_count {
                            visited_facts[index] = Some(fact);
                        }
                    }
                } else if let Some(place) = atom_argument(fact_name, "loc") {
                    if let Some(&index) = node_by_name.get(place) {
                        location_facts[index] = Some(fact);
                    }
                }
            }
        }

        let mut final_visited_facts = Vec::with_capacity(customer_count);
        for (index, fact) in visited_facts.into_iter().enumerate() {
            final_visited_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                format!(
                    "llm_001 could not resolve the visited fact for customer index {}",
                    index
                )
                .into()
            })?);
        }

        let mut final_location_facts = Vec::with_capacity(node_count);
        for (index, fact) in location_facts.into_iter().enumerate() {
            final_location_facts.push(fact.ok_or_else(|| -> HeuristicBuildError {
                format!(
                    "llm_001 could not resolve the location fact for node index {}",
                    index
                )
                .into()
            })?);
        }

        let mut direct_cost = vec![INF; node_count * node_count];
        let mut via_depot_cost = vec![INF; node_count * node_count];
        for node in 0..node_count {
            direct_cost[node * node_count + node] = 0.0;
            via_depot_cost[node * node_count + node] = 0.0;
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
                        (node_by_name.get(tokens[1]), node_by_name.get(tokens[2]))
                    {
                        let slot = from * node_count + to;
                        direct_cost[slot] = direct_cost[slot].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(tokens[1]), node_by_name.get(tokens[2]))
                    {
                        let slot = from * node_count + to;
                        via_depot_cost[slot] = via_depot_cost[slot].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = node_by_name.get(tokens[1]) {
                        let slot = from * node_count + depot;
                        direct_cost[slot] = direct_cost[slot].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Reconstruct missing via-depot transitions from the two actual legs.
        for from in 0..customer_count {
            for to in 0..customer_count {
                if from == to {
                    continue;
                }
                let first = direct_cost[from * node_count + depot];
                let second = direct_cost[depot * node_count + to];
                if first.is_finite() && second.is_finite() {
                    let slot = from * node_count + to;
                    via_depot_cost[slot] = via_depot_cost[slot].min(first + second);
                }
            }
        }

        let numeric_count = task.numeric_variables().len();
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut max_vehicles_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> =
            (0..customer_count).map(|_| None).collect();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name().trim();
            if numeric_name_is(name, "load()") {
                load_id = Some(id);
            } else if numeric_name_is(name, "vehicles()") {
                vehicles_id = Some(id);
            } else if numeric_name_is(name, "capacity()") {
                capacity_id = Some(id);
            } else if numeric_name_is(name, "max_vehicles()") {
                max_vehicles_id = Some(id);
            } else {
                for (customer, customer_name) in customer_names.iter().enumerate() {
                    let exact = format!("demand({})", customer_name);
                    if numeric_name_is(name, &exact) {
                        demand_ids[customer] = Some(id);
                        break;
                    }
                }
            }
        }

        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0);
        let max_vehicles = max_vehicles_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value >= 0.0);

        let mut demands = vec![0.0; customer_count];
        let mut all_demands_available = true;
        for customer in 0..customer_count {
            if let Some(id) = demand_ids[customer] {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value >= 0.0 {
                        demands[customer] = *value;
                        continue;
                    }
                }
            }
            all_demands_available = false;
        }

        // Capacity reasoning is enabled only when every demand, capacity, and
        // current load are available under exact names. Otherwise demands are
        // deliberately treated as zero and the routing estimate remains active.
        let capacity = if all_demands_available && load_id.is_some() {
            capacity
        } else {
            None
        };
        if capacity.is_none() {
            demands.fill(0.0);
        }

        Ok(Self {
            customer_count,
            depot,
            visited_facts: final_visited_facts,
            location_facts: final_location_facts,
            direct_cost,
            via_depot_cost,
            demands,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
            numeric_count,
        })
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.direct_cost[from * (self.customer_count + 1) + to]
    }

    fn via_cost(&self, from: usize, to: usize) -> f64 {
        self.via_depot_cost[from * (self.customer_count + 1) + to]
    }

    fn relaxed_edge(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }
        let direct = self.cost(from, to);
        let via = self.via_cost(from, to);
        direct.min(via)
    }

    fn routing_lower_bound(&self, current: usize, unvisited: &[bool]) -> f64 {
        let mut active: Vec<usize> = Vec::with_capacity(self.customer_count + 2);
        active.push(current);
        for customer in 0..self.customer_count {
            if unvisited[customer] && customer != current {
                active.push(customer);
            }
        }
        if !active.iter().any(|node| *node == self.depot) {
            active.push(self.depot);
        }

        if active.len() <= 1 {
            return 0.0;
        }

        // Undirected MST relaxation over the current endpoint, all remaining
        // customers, and the required final depot.
        let count = active.len();
        let mut in_tree = vec![false; count];
        let mut best = vec![INF; count];
        best[0] = 0.0;
        let mut mst: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = INF;
            for i in 0..count {
                if !in_tree[i] && best[i] < selected_cost {
                    selected = Some(i);
                    selected_cost = best[i];
                }
            }

            let Some(selected) = selected else {
                return 0.0;
            };
            if !selected_cost.is_finite() {
                return 0.0;
            }

            in_tree[selected] = true;
            mst += selected_cost;
            let from = active[selected];

            for other in 0..count {
                if in_tree[other] {
                    continue;
                }
                let to = active[other];
                let edge = self.relaxed_edge(from, to)
                    .min(self.relaxed_edge(to, from));
                if edge < best[other] {
                    best[other] = edge;
                }
            }
        }

        // Every node except the final depot must still have an outgoing leg.
        // This directed assignment-style relaxation complements the MST.
        let mut outgoing: f64 = 0.0;
        for &from in &active {
            if from == self.depot && current != self.depot {
                continue;
            }
            let mut cheapest = INF;
            for &to in &active {
                if from != to {
                    cheapest = cheapest.min(self.relaxed_edge(from, to));
                }
            }
            if cheapest.is_finite() {
                outgoing += cheapest;
            }
        }

        mst.max(outgoing)
    }

    fn greedy_completion(
        &self,
        current: usize,
        unvisited: &[bool],
        initial_load: f64,
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|value| **value).count();
        let mut at = current;
        let mut load = if current == self.depot { 0.0 } else { initial_load };
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = INF;

            for customer in 0..self.customer_count {
                if !remaining[customer] {
                    continue;
                }
                let feasible = match self.capacity {
                    Some(capacity) => load + self.demands[customer] <= capacity + 1.0e-9,
                    None => true,
                };
                if !feasible {
                    continue;
                }

                let edge = self.cost(at, customer);
                if edge < best_cost {
                    best_cost = edge;
                    best_customer = Some(customer);
                }
            }

            if let Some(customer) = best_customer {
                total += best_cost;
                at = customer;
                load += self.demands[customer];
                remaining[customer] = false;
                remaining_count -= 1;
                continue;
            }

            // No customer fits on the current route. Select the cheapest next
            // customer reached through the depot, which corresponds exactly to
            // visit-via-depot in this domain.
            let mut next_customer: Option<usize> = None;
            let mut next_cost = INF;
            for customer in 0..self.customer_count {
                if !remaining[customer] {
                    continue;
                }
                let edge = if at == self.depot {
                    self.cost(self.depot, customer)
                } else {
                    self.via_cost(at, customer)
                };
                if edge < next_cost {
                    next_cost = edge;
                    next_customer = Some(customer);
                }
            }

            let Some(customer) = next_customer else {
                break;
            };
            if !next_cost.is_finite() {
                break;
            }

            total += next_cost;
            at = customer;
            load = self.demands[customer];
            remaining[customer] = false;
            remaining_count -= 1;
        }

        let return_cost = self.cost(at, self.depot);
        if return_cost.is_finite() {
            total += return_cost;
        }
        total
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

        let mut current = self.depot;
        for (node, fact) in self.location_facts.iter().enumerate() {
            if fact.is_hold(view) {
                current = node;
                break;
            }
        }

        let mut unvisited = vec![false; self.customer_count];
        let mut total_remaining_demand: f64 = 0.0;
        for customer in 0..self.customer_count {
            if !self.visited_facts[customer].is_hold(view) {
                unvisited[customer] = true;
                total_remaining_demand += self.demands[customer];
            }
        }

        let mut current_load: f64 = 0.0;
        let mut current_vehicles: f64 = 0.0;
        if self.load_id.is_some() || self.vehicles_id.is_some() {
            let mut numeric_values = vec![0.0; self.numeric_count];
            view.fill_numeric(&mut numeric_values).map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

            if let Some(id) = self.load_id {
                if let Some(value) = numeric_values.get(id) {
                    if value.is_finite() {
                        current_load = *value;
                    }
                }
            }
            if let Some(id) = self.vehicles_id {
                if let Some(value) = numeric_values.get(id) {
                    if value.is_finite() {
                        current_vehicles = *value;
                    }
                }
            }
        }

        let lower = self.routing_lower_bound(current, &unvisited);
        let greedy = self.greedy_completion(current, &unvisited, current_load);

        // A definite aggregate-capacity violation of the vehicle limit marks a
        // dead-end-like state. This check is used only when all exact resource
        // fluents survived translation.
        if let (Some(capacity), Some(max_vehicles)) =
            (self.capacity, self.max_vehicles)
        {
            let available_on_current = if current == self.depot {
                capacity
            } else {
                (capacity - current_load).max(0.0)
            };
            let demand_after_current =
                (total_remaining_demand - available_on_current).max(0.0);
            let additional_routes = (demand_after_current / capacity).ceil();
            if current_vehicles + additional_routes > max_vehicles + 1.0e-9 {
                return Ok(1.0e12 + greedy.max(lower));
            }
        }

        // Blend a stable graph relaxation with a capacity-aware constructive
        // completion estimate. The heuristic is intentionally non-admissible.
        let h = if greedy.is_finite() {
            0.25 * lower + 0.75 * greedy.max(lower)
        } else {
            lower
        };

        Ok(h.max(0.0))
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
