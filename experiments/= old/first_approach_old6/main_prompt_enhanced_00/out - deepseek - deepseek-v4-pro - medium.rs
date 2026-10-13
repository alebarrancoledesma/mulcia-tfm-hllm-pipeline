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
use std::collections::{BTreeSet, HashMap};

fn numeric_name_matches(name: &str, short: &str) -> bool {
    let name = name.trim();
    name == short || name == format!("PNE {}", short)
}

fn extract_fact_arg(name: &str, prefixes: &[&str]) -> Option<String> {
    let name = name.trim();
    for prefix in prefixes {
        if name.starts_with(prefix) && name.ends_with(')') && name.len() > prefix.len() + 1 {
            return Some(name[prefix.len()..name.len() - 1].to_string());
        }
    }
    None
}

struct ScratchBuffers {
    prop: Vec<usize>,
    numeric: Vec<f64>,
    remaining: Vec<usize>,
    remaining2: Vec<usize>,
    order: Vec<usize>,
}

struct LlmHeuristic {
    n_customers: usize,
    depot: usize,
    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    demand: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    fallback_edge: f64,
    vehicle_penalty: f64,
    visited_fact: Vec<Option<(usize, usize)>>,
    loc_var_id: usize,
    loc_value_to_node: Vec<usize>,
    load_var_id: Option<usize>,
    vehicles_var_id: Option<usize>,
    scratch: RefCell<ScratchBuffers>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let variables = task.variables();
        let numeric_variables = task.numeric_variables();
        let operators = task.get_operators();

        let mut loc_var_id: Option<usize> = None;
        let mut visited_var_id: Option<usize> = None;
        let mut loc_facts: Vec<(usize, String)> = Vec::new();
        let mut visited_facts: Vec<(usize, String)> = Vec::new();

        for (var_id, var) in variables.iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact).to_string();
                let name = name.trim();

                if let Some(arg) = extract_fact_arg(name, &["Atom loc(", "loc("]) {
                    if loc_var_id.is_none() {
                        loc_var_id = Some(var_id);
                    }
                    if loc_var_id == Some(var_id) {
                        loc_facts.push((value, arg));
                    }
                }

                if let Some(arg) = extract_fact_arg(name, &["Atom visited(", "visited("]) {
                    if visited_var_id.is_none() {
                        visited_var_id = Some(var_id);
                    }
                    if visited_var_id == Some(var_id) {
                        visited_facts.push((value, arg));
                    }
                }
            }
        }

        let loc_var_id = loc_var_id
            .ok_or_else(|| HeuristicBuildError::from("Missing loc proposition".to_string()))?;
        let visited_var_id = visited_var_id
            .ok_or_else(|| HeuristicBuildError::from("Missing visited proposition".to_string()))?;

        let mut place_names: BTreeSet<String> = BTreeSet::new();
        for (_, name) in &loc_facts {
            place_names.insert(name.clone());
        }

        let mut customer_names: BTreeSet<String> = BTreeSet::new();
        for (_, name) in &visited_facts {
            customer_names.insert(name.clone());
        }

        let depot_candidates: Vec<String> = place_names
            .difference(&customer_names)
            .cloned()
            .collect();
        if depot_candidates.is_empty() {
            return Err("Could not identify a depot location".to_string().into());
        }
        let depot_name = depot_candidates[0].clone();

        let mut sorted_customer_names: Vec<String> = customer_names.iter().cloned().collect();
        sorted_customer_names.sort();

        let n_customers = sorted_customer_names.len();
        if n_customers == 0 {
            return Err("No customers found".to_string().into());
        }
        let n_nodes = n_customers + 1;
        let depot = 0usize;

        let mut name_to_node: HashMap<String, usize> = HashMap::new();
        name_to_node.insert(depot_name.clone(), depot);
        for (idx, name) in sorted_customer_names.iter().enumerate() {
            name_to_node.insert(name.clone(), idx + 1);
        }

        let loc_domain_size = variables
            .get(loc_var_id)
            .map(|v| v.domain_size())
            .unwrap_or(0);
        let mut loc_value_to_node = vec![usize::MAX; loc_domain_size];
        for (value, name) in &loc_facts {
            if let Some(&node) = name_to_node.get(name.as_str()) {
                if *value < loc_value_to_node.len() {
                    loc_value_to_node[*value] = node;
                }
            }
        }

        let mut visited_fact: Vec<Option<(usize, usize)>> = vec![None; n_nodes];
        for (value, name) in &visited_facts {
            if let Some(&node) = name_to_node.get(name.as_str()) {
                if node >= 1 && node <= n_customers {
                    visited_fact[node] = Some((visited_var_id, *value));
                }
            }
        }
        for idx in 1..=n_customers {
            if visited_fact[idx].is_none() {
                return Err(format!("Missing visited fact for customer {}", sorted_customer_names[idx - 1]).into());
            }
        }

        let mut load_var_id: Option<usize> = None;
        let mut vehicles_var_id: Option<usize> = None;
        let mut capacity_var_id: Option<usize> = None;
        let mut max_vehicles_var_id: Option<usize> = None;

        for (id, var) in numeric_variables.iter().enumerate() {
            let name = var.name().trim().to_string();
            if numeric_name_matches(&name, "load()") {
                load_var_id = Some(id);
            } else if numeric_name_matches(&name, "vehicles()") {
                vehicles_var_id = Some(id);
            } else if numeric_name_matches(&name, "capacity()") {
                capacity_var_id = Some(id);
            } else if numeric_name_matches(&name, "max_vehicles()") {
                max_vehicles_var_id = Some(id);
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();

        let capacity = capacity_var_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(f64::MAX);

        let max_vehicles = max_vehicles_var_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(f64::MAX);

        let mut demand = vec![0.0; n_nodes];

        for (idx, customer_name) in sorted_customer_names.iter().enumerate() {
            let short = format!("demand({})", customer_name);
            for (id, var) in numeric_variables.iter().enumerate() {
                let name = var.name().trim().to_string();
                if numeric_name_matches(&name, &short) && id < initial_numeric.len() {
                    let v = initial_numeric[id];
                    if v.is_finite() && v >= 0.0 {
                        demand[idx + 1] = v;
                    }
                }
            }
        }

        if let Some(load_id) = load_var_id {
            for op in operators.iter() {
                let op_name = op.name();
                let tokens: Vec<&str> = op_name.split_whitespace().collect();
                let to_name = match tokens.first().copied() {
                    Some("visit") if tokens.len() == 3 => Some(tokens[2]),
                    Some("visit-via-depot") if tokens.len() == 3 => Some(tokens[2]),
                    _ => None,
                };
                if let Some(to_name) = to_name {
                    if let Some(&to_idx) = name_to_node.get(to_name) {
                        if to_idx >= 1 && to_idx <= n_customers {
                            for assign in op.assignment_effects().iter() {
                                if assign.affected_var_id() == load_id {
                                    let rhs = assign.var_id();
                                    if rhs < initial_numeric.len() {
                                        let v = initial_numeric[rhs];
                                        if v.is_finite() && v >= 0.0 {
                                            demand[to_idx] = v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut direct_cost = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
        let mut via_depot_cost = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
        for i in 0..n_nodes {
            direct_cost[i][i] = 0.0;
            via_depot_cost[i][i] = 0.0;
        }

        for op in operators.iter() {
            let op_name = op.name();
            let tokens: Vec<&str> = op_name.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) = (
                        name_to_node.get(tokens[1]),
                        name_to_node.get(tokens[2]),
                    ) {
                        direct_cost[from][to] = direct_cost[from][to].min(cost);
                    }
                }
                "return-to-depot" if tokens.len() == 2 => {
                    if let Some(&from) = name_to_node.get(tokens[1]) {
                        direct_cost[from][depot] = direct_cost[from][depot].min(cost);
                    }
                }
                "visit-via-depot" if tokens.len() == 3 => {
                    if let (Some(&from), Some(&to)) = (
                        name_to_node.get(tokens[1]),
                        name_to_node.get(tokens[2]),
                    ) {
                        via_depot_cost[from][to] = via_depot_cost[from][to].min(cost);
                    }
                }
                _ => {}
            }
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i != j && !via_depot_cost[i][j].is_finite() {
                    let di = direct_cost[i][depot];
                    let dj = direct_cost[depot][j];
                    if di.is_finite() && dj.is_finite() {
                        via_depot_cost[i][j] = di + dj;
                    }
                }
            }
        }

        let mut max_edge = 1.0f64;
        for row in direct_cost.iter().chain(via_depot_cost.iter()) {
            for &v in row.iter() {
                if v.is_finite() && v > max_edge {
                    max_edge = v;
                }
            }
        }

        let fallback_edge = max_edge * 2.0 + 1.0;
        let vehicle_penalty = fallback_edge * n_customers as f64 * 2.0 + 1.0;

        let scratch = RefCell::new(ScratchBuffers {
            prop: vec![0; variables.len()],
            numeric: vec![0.0; numeric_variables.len()],
            remaining: Vec::with_capacity(n_customers),
            remaining2: Vec::with_capacity(n_customers),
            order: Vec::with_capacity(n_customers),
        });

        Ok(Self {
            n_customers,
            depot,
            direct_cost,
            via_depot_cost,
            demand,
            capacity,
            max_vehicles,
            fallback_edge,
            vehicle_penalty,
            visited_fact,
            loc_var_id,
            loc_value_to_node,
            load_var_id,
            vehicles_var_id,
            scratch,
        })
    }

    fn return_cost(&self, i: usize) -> f64 {
        if i == self.depot {
            return 0.0;
        }
        let c = self.direct_cost[i][self.depot];
        if c.is_finite() {
            c
        } else {
            self.fallback_edge
        }
    }

    fn order_cost(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let mut best = self.direct_cost[i][j];
        if !best.is_finite() && i != self.depot && j != self.depot {
            best = self.via_depot_cost[i][j];
        }
        if !best.is_finite() {
            best = self.fallback_edge;
        }
        best
    }

    fn via_edge_cost_opt(&self, i: usize, j: usize) -> Option<f64> {
        if i == j {
            return Some(0.0);
        }
        let mut best = self.via_depot_cost[i][j];
        if !best.is_finite() {
            let di = self.direct_cost[i][self.depot];
            let dj = self.direct_cost[self.depot][j];
            if di.is_finite() && dj.is_finite() {
                best = di + dj;
            }
        }
        if best.is_finite() {
            Some(best)
        } else {
            None
        }
    }

    fn via_cost(&self, i: usize, j: usize) -> f64 {
        self.via_edge_cost_opt(i, j).unwrap_or_else(|| {
            let fallback = self.order_cost(i, j)
                + self.return_cost(i)
                + self.order_cost(self.depot, j);
            if fallback.is_finite() {
                fallback
            } else {
                self.fallback_edge
            }
        })
    }

    fn append_nearest_order(
        &self,
        start: usize,
        remaining: &mut Vec<usize>,
        order: &mut Vec<usize>,
    ) {
        order.clear();
        let mut cur = start;
        while !remaining.is_empty() {
            let mut best_idx = 0usize;
            let mut best_score = f64::INFINITY;

            for idx in 0..remaining.len() {
                let c = remaining[idx];
                let d = self.order_cost(cur, c);
                let dep = self.return_cost(c);
                let score = d + 0.35 * dep;
                if score < best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }

            let c = remaining.swap_remove(best_idx);
            order.push(c);
            cur = c;
        }
    }

    fn score_order(
        &self,
        start: usize,
        init_load: f64,
        init_vehicles: f64,
        order: &[usize],
    ) -> f64 {
        let mut cur = start;
        let mut load = init_load.max(0.0);
        let mut vehicles = init_vehicles.max(0.0);
        let mut total = 0.0;

        for &c in order {
            let dem = self.demand[c];
            let direct_ok = self.direct_cost[cur][c].is_finite()
                && load + dem <= self.capacity;

            if direct_ok || cur == self.depot {
                total += self.order_cost(cur, c);
                load += dem;
            } else {
                total += self.via_cost(cur, c);
                if vehicles + 1.0 > self.max_vehicles {
                    let excess = (vehicles + 1.0 - self.max_vehicles).max(0.0);
                    total += self.vehicle_penalty * excess;
                }
                vehicles += 1.0;
                load = dem;
            }
            cur = c;
        }

        total += self.return_cost(cur);

        if !total.is_finite() {
            total = self.fallback_edge * order.len().max(1) as f64 * 2.0;
        }
        total.max(0.0)
    }

    fn estimate_greedy(
        &self,
        start: usize,
        init_load: f64,
        init_vehicles: f64,
        remaining: &mut Vec<usize>,
    ) -> f64 {
        let mut cur = start;
        let mut load = init_load.max(0.0);
        let mut vehicles = init_vehicles.max(0.0);
        let mut total = 0.0;

        while !remaining.is_empty() {
            let mut best_idx = 0usize;
            let mut best_cost = f64::INFINITY;
            let mut best_is_direct = false;

            for idx in 0..remaining.len() {
                let c = remaining[idx];
                let dem = self.demand[c];

                let direct_edge_available =
                    self.direct_cost[cur][c].is_finite() || cur == self.depot;
                let direct_possible =
                    direct_edge_available && load + dem <= self.capacity;
                let via_possible = cur != self.depot
                    && vehicles + 1.0 <= self.max_vehicles
                    && self.via_edge_cost_opt(cur, c).is_some();

                if direct_possible {
                    let cost = self.order_cost(cur, c) + 0.15 * self.return_cost(c);
                    if cost < best_cost {
                        best_cost = cost;
                        best_idx = idx;
                        best_is_direct = true;
                    }
                }

                if via_possible {
                    let cost = self.via_cost(cur, c) + 0.15 * self.return_cost(c);
                    if cost < best_cost {
                        best_cost = cost;
                        best_idx = idx;
                        best_is_direct = false;
                    }
                }
            }

            if !best_cost.is_finite() {
                for idx in 0..remaining.len() {
                    let c = remaining[idx];
                    let cost = self.order_cost(cur, c);
                    if cost < best_cost {
                        best_cost = cost;
                        best_idx = idx;
                        best_is_direct = true;
                    }
                }
            }

            if !best_cost.is_finite() {
                best_cost = self.fallback_edge;
                best_is_direct = true;
            }

            let c = remaining.swap_remove(best_idx);
            let dem = self.demand[c];

            if best_is_direct {
                total += self.order_cost(cur, c);
                load += dem;
            } else {
                total += self.via_cost(cur, c);
                if vehicles + 1.0 > self.max_vehicles {
                    let excess = (vehicles + 1.0 - self.max_vehicles).max(0.0);
                    total += self.vehicle_penalty * excess;
                }
                vehicles += 1.0;
                load = dem;
            }
            cur = c;
        }

        total += self.return_cost(cur);

        if !total.is_finite() {
            total = self.fallback_edge * self.n_customers.max(1) as f64 * 2.0;
        }
        total.max(0.0)
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

        let mut scratch = self.scratch.borrow_mut();

        registry
            .view(state)
            .fill_numeric(&mut scratch.numeric)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

        let _ = state.fill_state(registry, &mut scratch.prop);

        let cur = scratch
            .prop
            .get(self.loc_var_id)
            .and_then(|v| self.loc_value_to_node.get(*v))
            .copied()
            .unwrap_or(self.depot);

        let load = self
            .load_var_id
            .and_then(|id| scratch.numeric.get(id).copied())
            .unwrap_or(0.0)
            .max(0.0);

        let vehicles = self
            .vehicles_var_id
            .and_then(|id| scratch.numeric.get(id).copied())
            .unwrap_or(1.0)
            .max(0.0);

        scratch.remaining.clear();
        scratch.remaining2.clear();

        for c in 1..=self.n_customers {
            let visited = self.visited_fact[c]
                .map(|(var_id, val)| scratch.prop[var_id] == val)
                .unwrap_or(false);
            if !visited {
                scratch.remaining.push(c);
                scratch.remaining2.push(c);
            }
        }

        scratch.order.clear();
        self.append_nearest_order(cur, &mut scratch.remaining, &mut scratch.order);
        let h1 = self.score_order(cur, load, vehicles, &scratch.order);

        let h2 = self.estimate_greedy(cur, load, vehicles, &mut scratch.remaining2);

        let mut h = h1.min(h2);
        if !h.is_finite() {
            h = self.fallback_edge * self.n_customers.max(1) as f64 * 2.0;
        }
        if h < 0.0 {
            h = 0.0;
        }

        Ok(h)
    }

    fn heuristic_name(&self) -> &str {
        "llm_001"
    }
}

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
        return Err("`llm_001` does not accept arguments".to_string().into());
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
