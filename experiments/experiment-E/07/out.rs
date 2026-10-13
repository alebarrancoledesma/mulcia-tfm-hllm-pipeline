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

const INF: f64 = f64::MAX / 16.0;

struct LlmHeuristic {
    // Node zero is the depot; nodes 1.. are customers.
    node_names: Vec<String>,
    visited_facts: Vec<Option<ExplicitFact>>,
    location_facts: Vec<ExplicitFact>,

    // Direct travel costs reconstructed from grounded visit/return operators.
    direct_cost: Vec<Vec<f64>>,

    // Costs of visit-via-depot operators. Missing entries are reconstructed
    // from the two direct depot legs at evaluation time.
    via_cost: Vec<Vec<f64>>,

    // Optional resource information. Constant quantities are copied from the
    // task's initial numeric values, while load_id is read from each state.
    load_id: Option<usize>,
    capacity: f64,
    demands: Vec<f64>,
    has_capacity_data: bool,
}

fn atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let trimmed = name.trim();
    let prefix = format!("Atom {}(", predicate);
    if !trimmed.starts_with(&prefix) || !trimmed.ends_with(')') {
        return None;
    }
    Some(&trimmed[prefix.len()..trimmed.len() - 1])
}

fn operator_tokens(name: &str) -> Vec<&str> {
    name.split_whitespace()
        .map(|token| token.trim_matches(|c| c == '(' || c == ')'))
        .filter(|token| !token.is_empty())
        .collect()
}

fn exact_numeric_name(name: &str, bare: &str) -> bool {
    let trimmed = name.trim();
    trimmed == bare || trimmed == format!("PNE {}", bare)
}

fn parse_exact_unary_numeric_name<'a>(name: &'a str, function: &str) -> Option<&'a str> {
    let mut trimmed = name.trim();
    if let Some(rest) = trimmed.strip_prefix("PNE ") {
        trimmed = rest;
    }

    let prefix = format!("{}(", function);
    if !trimmed.starts_with(&prefix) || !trimmed.ends_with(')') {
        return None;
    }
    Some(&trimmed[prefix.len()..trimmed.len() - 1])
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        // The CVRP goals identify all customers through visited facts and the
        // depot through the required final location fact.
        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            let fact_name = task.get_fact_name(goal);

            if let Some(customer) = atom_argument(fact_name, "visited") {
                customer_names.push(customer.to_string());
            } else if let Some(place) = atom_argument(fact_name, "loc") {
                depot_name = Some(place.to_string());
            }
        }

        customer_names.sort();
        customer_names.dedup();

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            HeuristicBuildError::from("missing depot location goal fact".to_string())
        })?;

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "CVRP task has no visited-customer goals".to_string(),
            ));
        }

        let mut node_names = Vec::with_capacity(customer_names.len() + 1);
        node_names.push(depot_name);
        node_names.extend(customer_names);

        let mut node_index: HashMap<String, usize> = HashMap::new();
        for (index, name) in node_names.iter().enumerate() {
            node_index.insert(name.clone(), index);
        }

        let node_count = node_names.len();
        let mut visited_facts: Vec<Option<ExplicitFact>> =
            (0..node_count).map(|_| None).collect();
        let mut location_facts: Vec<Option<ExplicitFact>> =
            (0..node_count).map(|_| None).collect();

        // Resolve propositional fact IDs once. Truth values are identified by
        // semantic fact names rather than assuming a particular SAS value.
        for variable_id in 0..task.variables().len() {
            let domain_size = task.variables()[variable_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(place) = atom_argument(&fact_name, "loc") {
                    if let Some(&index) = node_index.get(place) {
                        location_facts[index] = Some(fact);
                        continue;
                    }
                }

                if let Some(customer) = atom_argument(&fact_name, "visited") {
                    if let Some(&index) = node_index.get(customer) {
                        if index != 0 {
                            visited_facts[index] = Some(fact);
                        }
                    }
                }
            }
        }

        let mut resolved_locations = Vec::with_capacity(node_count);
        for (index, fact) in location_facts.into_iter().enumerate() {
            resolved_locations.push(fact.ok_or_else(|| -> HeuristicBuildError {
                HeuristicBuildError::from(format!(
                    "missing location fact for routing node {}",
                    node_names[index]
                ))
            })?);
        }

        for index in 1..node_count {
            if visited_facts[index].is_none() {
                return Err(HeuristicBuildError::from(format!(
                    "missing visited fact for customer {}",
                    node_names[index]
                )));
            }
        }

        let mut direct_cost = vec![vec![INF; node_count]; node_count];
        let mut via_cost = vec![vec![INF; node_count]; node_count];
        direct_cost[0][0] = 0.0;

        // Grounded operators preserve all routing endpoints and provide metric
        // costs even when the translator folds away travel-cost PNEs.
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
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        direct_cost[from][to] = direct_cost[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = node_index.get(tokens[1]) {
                        direct_cost[from][0] = direct_cost[from][0].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_index.get(tokens[1]), node_index.get(tokens[2]))
                    {
                        via_cost[from][to] = via_cost[from][to].min(cost);
                    }
                }
                _ => {}
            }
        }

        // Routing costs are the essential signal. Require the direct arcs that
        // exist in this domain, but do not require optional resource PNEs.
        for customer in 1..node_count {
            if direct_cost[0][customer] >= INF || direct_cost[customer][0] >= INF {
                return Err(HeuristicBuildError::from(format!(
                    "could not reconstruct depot travel costs for {}",
                    node_names[customer]
                )));
            }
        }
        for from in 1..node_count {
            for to in 1..node_count {
                if from != to && direct_cost[from][to] >= INF {
                    return Err(HeuristicBuildError::from(format!(
                        "could not reconstruct direct travel cost {} -> {}",
                        node_names[from], node_names[to]
                    )));
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut demand_ids: Vec<Option<usize>> =
            (0..node_count).map(|_| None).collect();

        // Only exact numeric names are accepted. In particular, synthesized
        // derived expressions containing words such as capacity or demand are
        // deliberately ignored.
        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name();
            if exact_numeric_name(name, "load()") {
                load_id = Some(id);
            } else if exact_numeric_name(name, "capacity()") {
                capacity_id = Some(id);
            } else if let Some(customer) = parse_exact_unary_numeric_name(name, "demand") {
                if let Some(&index) = node_index.get(customer) {
                    if index != 0 {
                        demand_ids[index] = Some(id);
                    }
                }
            }
        }

        let capacity = capacity_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(f64::MAX);

        let mut demands = vec![0.0_f64; node_count];
        let mut all_demands_known = capacity < f64::MAX;
        for customer in 1..node_count {
            if let Some(id) = demand_ids[customer] {
                if let Some(value) = initial_numeric.get(id).copied() {
                    if value.is_finite() && value >= 0.0 {
                        demands[customer] = value;
                        continue;
                    }
                }
            }
            all_demands_known = false;
        }

        let has_capacity_data = all_demands_known && load_id.is_some();

        Ok(Self {
            node_names,
            visited_facts,
            location_facts: resolved_locations,
            direct_cost,
            via_cost,
            load_id,
            capacity,
            demands,
            has_capacity_data,
        })
    }

    fn undirected_cost(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }
        self.direct_cost[from][to].min(self.direct_cost[to][from])
    }

    fn via_or_reconstructed(&self, from: usize, to: usize) -> f64 {
        let grounded = self.via_cost[from][to];
        if grounded < INF {
            grounded
        } else {
            self.direct_cost[from][0] + self.direct_cost[0][to]
        }
    }

    fn mst_bound(&self, current: usize, unvisited: &[bool]) -> f64 {
        let mut active: Vec<usize> = Vec::new();
        active.push(0);
        if current != 0 {
            active.push(current);
        }
        for customer in 1..self.node_names.len() {
            if unvisited[customer] && customer != current {
                active.push(customer);
            }
        }

        if active.len() <= 1 {
            return 0.0;
        }

        let count = active.len();
        let mut used = vec![false; count];
        let mut best = vec![INF; count];
        best[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut selected: Option<usize> = None;
            let mut selected_cost = INF;
            for index in 0..count {
                if !used[index] && best[index] < selected_cost {
                    selected = Some(index);
                    selected_cost = best[index];
                }
            }

            let selected = match selected {
                Some(index) => index,
                None => return total,
            };
            used[selected] = true;
            total += selected_cost;

            for other in 0..count {
                if !used[other] {
                    let edge = self.undirected_cost(active[selected], active[other]);
                    if edge < best[other] {
                        best[other] = edge;
                    }
                }
            }
        }

        total
    }

    fn greedy_completion(
        &self,
        start: usize,
        initial_load: f64,
        unvisited: &[bool],
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|value| **value).count();
        let mut current = start;
        let mut load = initial_load.max(0.0);
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_leg = INF;
            let mut best_score = INF;
            let mut best_uses_depot = false;

            for customer in 1..self.node_names.len() {
                if !remaining[customer] {
                    continue;
                }

                let demand = self.demands[customer];
                let direct_feasible = !self.has_capacity_data
                    || current == 0
                    || load + demand <= self.capacity + 1.0e-9;

                let mut leg = INF;
                let mut uses_depot = false;
                if direct_feasible {
                    leg = self.direct_cost[current][customer];
                }

                if current != 0 {
                    let via = self.via_or_reconstructed(current, customer);
                    if via < leg {
                        leg = via;
                        uses_depot = true;
                    }
                }

                if leg >= INF {
                    continue;
                }

                // One-step lookahead discourages selecting a cheap customer
                // that leaves an expensive isolated remainder.
                let mut continuation = self.direct_cost[customer][0];
                for next in 1..self.node_names.len() {
                    if next != customer && remaining[next] {
                        continuation = continuation.min(self.direct_cost[customer][next]);
                    }
                }
                let score = leg + 0.30 * continuation;

                if score < best_score {
                    best_score = score;
                    best_leg = leg;
                    best_customer = Some(customer);
                    best_uses_depot = uses_depot;
                }
            }

            let customer = match best_customer {
                Some(customer) => customer,
                None => {
                    // This should not occur in a well-formed reachable state.
                    // Fall back to an unconstrained via-depot choice rather
                    // than returning an infinite or negative heuristic.
                    let mut fallback: Option<usize> = None;
                    let mut fallback_cost = INF;
                    for candidate in 1..self.node_names.len() {
                        if remaining[candidate] {
                            let cost = if current == 0 {
                                self.direct_cost[0][candidate]
                            } else {
                                self.via_or_reconstructed(current, candidate)
                            };
                            if cost < fallback_cost {
                                fallback = Some(candidate);
                                fallback_cost = cost;
                            }
                        }
                    }
                    match fallback {
                        Some(candidate) => {
                            best_leg = fallback_cost;
                            best_uses_depot = current != 0;
                            candidate
                        }
                        None => break,
                    }
                }
            };

            total += best_leg;
            if self.has_capacity_data {
                if current == 0 || best_uses_depot {
                    load = self.demands[customer];
                } else {
                    load += self.demands[customer];
                }
            }

            current = customer;
            remaining[customer] = false;
            remaining_count -= 1;
        }

        if current != 0 {
            total += self.direct_cost[current][0];
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut current: Option<usize> = None;
        for (index, fact) in self.location_facts.iter().enumerate() {
            if fact.is_hold(registry.view(state)) {
                current = Some(index);
                break;
            }
        }
        let current = current.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "no CVRP location fact holds in the evaluated state".to_string(),
            )
        })?;

        let mut unvisited = vec![false; self.node_names.len()];
        for customer in 1..self.node_names.len() {
            let visited = self.visited_facts[customer]
                .as_ref()
                .map(|fact| fact.is_hold(registry.view(state)))
                .unwrap_or(false);
            unvisited[customer] = !visited;
        }

        let mut current_load: f64 = 0.0;
        if self.has_capacity_data {
            if let Some(load_id) = self.load_id {
                let mut numeric_values = vec![0.0_f64; task.numeric_variables().len()];
                registry
                    .view(state)
                    .fill_numeric(&mut numeric_values)
                    .map_err(|err| {
                        EvaluationError::ComputationFailed(format!("{err:?}"))
                    })?;
                if let Some(value) = numeric_values.get(load_id).copied() {
                    if value.is_finite() {
                        current_load = value.max(0.0);
                    }
                }
            }
        }

        let mst = self.mst_bound(current, &unvisited);
        let greedy = self.greedy_completion(current, current_load, &unvisited);

        // The MST supplies a stable global connectivity estimate, while the
        // dominant greedy term captures route order, the required final depot
        // return, and capacity-triggered depot transitions when resource PNEs
        // survived translation.
        let h = (0.20 * mst + 0.80 * greedy).max(0.0);
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
