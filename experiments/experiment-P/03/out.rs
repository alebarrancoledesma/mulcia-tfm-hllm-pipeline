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

fn atom_argument<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
    let name = name.trim();
    let prefix = format!("Atom {}(", predicate);
    let rest = name.strip_prefix(&prefix)?;
    let argument = rest.strip_suffix(')')?;

    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.trim())
    }
}

fn exact_pne_argument<'a>(name: &'a str, function: &str) -> Option<&'a str> {
    let name = name.trim();
    let prefix = format!("PNE {}(", function);
    let rest = name.strip_prefix(&prefix)?;
    let argument = rest.strip_suffix(')')?;

    if argument.is_empty() || argument.contains(',') {
        None
    } else {
        Some(argument.trim())
    }
}

fn cleaned_operator_tokens(name: &str) -> Vec<&str> {
    name.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect()
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_facts: Vec<(usize, usize)>,
    customer_loc_facts: Vec<(usize, usize)>,
    depot_loc_fact: (usize, usize),

    // Node zero is the depot; customer i is node i + 1.
    distances: Vec<Vec<f64>>,

    // Resource information is optional because the translator may fold away
    // capacity, demand, and max_vehicles PNEs.
    demands: Option<Vec<f64>>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_names: Vec<String> = Vec::new();
        let mut depot_name: Option<String> = None;

        // In this domain the visited goals identify all customers, while the
        // unique loc goal identifies the depot.
        for goal_id in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_id);
            let fact_name = task.get_fact_name(&fact);

            if let Some(customer) = atom_argument(fact_name, "visited") {
                customer_names.push(customer.to_string());
            } else if let Some(place) = atom_argument(fact_name, "loc") {
                depot_name = Some(place.to_string());
            }
        }

        if customer_names.is_empty() {
            return Err(HeuristicBuildError::from(
                "llm_001 could not identify any visited customer goals".to_string(),
            ));
        }

        let depot_name = depot_name.ok_or_else(|| -> HeuristicBuildError {
            "llm_001 could not identify the depot loc goal"
                .to_string()
                .into()
        })?;

        let mut visited_by_name: HashMap<String, (usize, usize)> = HashMap::new();
        let mut loc_by_name: HashMap<String, (usize, usize)> = HashMap::new();

        // Explicit variable names are generic, so inspect semantic fact names.
        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(customer) = atom_argument(fact_name, "visited") {
                    visited_by_name.insert(customer.to_string(), (var_id, value));
                }
                if let Some(place) = atom_argument(fact_name, "loc") {
                    loc_by_name.insert(place.to_string(), (var_id, value));
                }
            }
        }

        let depot_loc_fact = loc_by_name
            .get(&depot_name)
            .copied()
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not resolve the depot loc fact"
                    .to_string()
                    .into()
            })?;

        let mut visited_facts: Vec<(usize, usize)> = Vec::with_capacity(customer_names.len());
        let mut customer_loc_facts: Vec<(usize, usize)> =
            Vec::with_capacity(customer_names.len());

        for customer in &customer_names {
            visited_facts.push(
                visited_by_name
                    .get(customer)
                    .copied()
                    .ok_or_else(|| -> HeuristicBuildError {
                        format!("llm_001 could not resolve visited fact for {customer}").into()
                    })?,
            );
            customer_loc_facts.push(
                loc_by_name
                    .get(customer)
                    .copied()
                    .ok_or_else(|| -> HeuristicBuildError {
                        format!("llm_001 could not resolve loc fact for {customer}").into()
                    })?,
            );
        }

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        node_by_name.insert(depot_name.clone(), 0);
        for (customer_id, customer) in customer_names.iter().enumerate() {
            node_by_name.insert(customer.clone(), customer_id + 1);
        }

        let node_count = customer_names.len() + 1;
        let mut distances = vec![vec![INF; node_count]; node_count];
        for (node, row) in distances.iter_mut().enumerate() {
            row[node] = 0.0;
        }

        // Reconstruct routing costs from grounded actions. This avoids relying
        // on names of travel-cost PNEs, which may be translated away.
        for op in task.get_operators() {
            let tokens = cleaned_operator_tokens(op.name());
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(tokens[1]), node_by_name.get(tokens[2]))
                    {
                        distances[from][to] = distances[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = node_by_name.get(tokens[1]) {
                        distances[from][0] = distances[from][0].min(cost);
                    }
                }
                _ => {}
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut max_vehicles: Option<f64> = None;
        let mut demand_by_name: HashMap<String, f64> = HashMap::new();

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            let name = variable.name().trim();

            if name == "PNE load()" {
                load_id = Some(id);
            } else if name == "PNE vehicles()" {
                vehicles_id = Some(id);
            } else if name == "PNE capacity()" {
                if let Some(&value) = initial_numeric.get(id) {
                    if value.is_finite() && value > 0.0 {
                        capacity = Some(value);
                    }
                }
            } else if name == "PNE max_vehicles()" {
                if let Some(&value) = initial_numeric.get(id) {
                    if value.is_finite() {
                        max_vehicles = Some(value);
                    }
                }
            } else if let Some(customer) = exact_pne_argument(name, "demand") {
                if let Some(&value) = initial_numeric.get(id) {
                    if value.is_finite() && value >= 0.0 {
                        demand_by_name.insert(customer.to_string(), value);
                    }
                }
            }
        }

        // Enable demand-aware reasoning only when every customer demand was
        // retained under its exact PNE name.
        let demands = if demand_by_name.len() >= customer_names.len() {
            let mut values: Vec<f64> = Vec::with_capacity(customer_names.len());
            let mut complete = true;
            for customer in &customer_names {
                if let Some(&demand) = demand_by_name.get(customer) {
                    values.push(demand);
                } else {
                    complete = false;
                    break;
                }
            }
            if complete { Some(values) } else { None }
        } else {
            None
        };

        Ok(Self {
            customer_names,
            visited_facts,
            customer_loc_facts,
            depot_loc_fact,
            distances,
            demands,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
        })
    }

    fn undirected_distance(&self, a: usize, b: usize) -> f64 {
        self.distances[a][b].min(self.distances[b][a])
    }

    fn mst_bound(&self, current_node: usize, unvisited: &[bool]) -> f64 {
        let mut terminals: Vec<usize> = Vec::with_capacity(unvisited.len() + 2);
        terminals.push(0);

        if current_node != 0 {
            terminals.push(current_node);
        }

        for (customer_id, &is_unvisited) in unvisited.iter().enumerate() {
            if is_unvisited {
                terminals.push(customer_id + 1);
            }
        }

        if terminals.len() <= 1 {
            return 0.0;
        }

        let count = terminals.len();
        let mut used = vec![false; count];
        let mut key = vec![INF; count];
        key[0] = 0.0;
        let mut total: f64 = 0.0;

        for _ in 0..count {
            let mut best_index: Option<usize> = None;
            let mut best_key = INF;

            for i in 0..count {
                if !used[i] && key[i] < best_key {
                    best_key = key[i];
                    best_index = Some(i);
                }
            }

            let Some(chosen) = best_index else {
                return INF;
            };

            used[chosen] = true;
            total += key[chosen];

            for other in 0..count {
                if !used[other] {
                    let edge = self.undirected_distance(
                        terminals[chosen],
                        terminals[other],
                    );
                    if edge < key[other] {
                        key[other] = edge;
                    }
                }
            }
        }

        total
    }

    fn greedy_completion(
        &self,
        current_node: usize,
        unvisited: &[bool],
        initial_load: f64,
        use_capacity: bool,
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut remaining_count = remaining.iter().filter(|&&x| x).count();
        let mut node = current_node;
        let mut load = if use_capacity { initial_load.max(0.0) } else { 0.0 };
        let mut total: f64 = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = INF;

            for customer_id in 0..remaining.len() {
                if !remaining[customer_id] {
                    continue;
                }

                let demand_fits = if use_capacity {
                    let demand = self.demands.as_ref().unwrap()[customer_id];
                    load + demand <= self.capacity.unwrap() + 1.0e-9
                } else {
                    true
                };

                if !demand_fits {
                    continue;
                }

                let edge = self.distances[node][customer_id + 1];
                if edge < best_cost {
                    best_cost = edge;
                    best_customer = Some(customer_id);
                }
            }

            if let Some(customer_id) = best_customer {
                total += best_cost;
                node = customer_id + 1;
                remaining[customer_id] = false;
                remaining_count -= 1;

                if use_capacity {
                    load += self.demands.as_ref().unwrap()[customer_id];
                }
                continue;
            }

            // A new route is represented in the PDDL by visit-via-depot.
            // Its metric cost is exactly current->depot plus depot->customer,
            // so it can be simulated without parsing folded via-depot PNEs.
            if node == 0 {
                return INF;
            }

            let return_cost = self.distances[node][0];
            if !return_cost.is_finite() {
                return INF;
            }
            total += return_cost;
            node = 0;
            load = 0.0;
        }

        if node != 0 {
            let return_cost = self.distances[node][0];
            if !return_cost.is_finite() {
                return INF;
            }
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut propositional_values = vec![0usize; task.variables().len()];
        state.fill_state(registry, &mut propositional_values);

        let mut unvisited = vec![false; self.customer_names.len()];
        for (customer_id, &(var_id, value)) in self.visited_facts.iter().enumerate() {
            unvisited[customer_id] = propositional_values[var_id] != value;
        }

        let mut current_node: Option<usize> = None;
        if propositional_values[self.depot_loc_fact.0] == self.depot_loc_fact.1 {
            current_node = Some(0);
        } else {
            for (customer_id, &(var_id, value)) in
                self.customer_loc_facts.iter().enumerate()
            {
                if propositional_values[var_id] == value {
                    current_node = Some(customer_id + 1);
                    break;
                }
            }
        }

        let current_node = current_node.ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "llm_001 could not identify the current vehicle location".to_string(),
            )
        })?;

        let need_numeric = self.load_id.is_some() || self.vehicles_id.is_some();
        let mut numeric_values: Vec<f64> = Vec::new();
        if need_numeric {
            numeric_values.resize(task.numeric_variables().len(), 0.0);
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
        }

        let current_load = self
            .load_id
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(0.0);
        let current_vehicles = self
            .vehicles_id
            .and_then(|id| numeric_values.get(id).copied());

        let use_capacity = self.demands.is_some()
            && self.capacity.is_some()
            && self.load_id.is_some();

        if use_capacity {
            let capacity = self.capacity.unwrap();
            let demands = self.demands.as_ref().unwrap();
            let mut remaining_demand: f64 = 0.0;

            for (customer_id, &is_unvisited) in unvisited.iter().enumerate() {
                if is_unvisited {
                    let demand = demands[customer_id];
                    if demand > capacity + 1.0e-9 {
                        return Ok(INF);
                    }
                    remaining_demand += demand;
                }
            }

            // Aggregate capacity gives a sound lower bound on the number of
            // additional routes. If even this bound exceeds the exact vehicle
            // limit, the state cannot reach a goal.
            if let (Some(max_vehicles), Some(vehicles)) =
                (self.max_vehicles, current_vehicles)
            {
                let residual = if current_node == 0 {
                    capacity
                } else {
                    (capacity - current_load).max(0.0)
                };
                let excess = (remaining_demand - residual).max(0.0);
                let additional_routes = if excess <= 1.0e-9 {
                    0.0
                } else {
                    (excess / capacity).ceil()
                };

                if vehicles + additional_routes > max_vehicles + 1.0e-9 {
                    return Ok(INF);
                }
            }
        }

        let mst = self.mst_bound(current_node, &unvisited);
        let greedy = self.greedy_completion(
            current_node,
            &unvisited,
            current_load,
            use_capacity,
        );

        if !mst.is_finite() || !greedy.is_finite() {
            return Ok(INF);
        }

        // The MST is a stable global routing signal; the capacity-aware
        // nearest-neighbour completion supplies a much more informative cost
        // estimate. A small MST contribution dampens greedy-route artifacts.
        let h = (0.85 * greedy + 0.15 * mst).max(0.0);
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
