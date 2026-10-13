use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Copy)]
struct FactRef {
    var_id: usize,
    value: usize,
}

struct LocFact {
    node: usize,
    fact: FactRef,
}

struct VisitedFact {
    node: usize,
    fact: FactRef,
}

struct ScratchBuffers {
    numeric: Vec<f64>,
    unvisited: Vec<usize>,
}

fn intern(name: &str, nodes: &mut Vec<String>, index: &mut HashMap<String, usize>) -> usize {
    if let Some(&idx) = index.get(name) {
        return idx;
    }
    let idx = nodes.len();
    nodes.push(name.to_string());
    index.insert(name.to_string(), idx);
    idx
}

fn extract_loc(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("Atom loc(") {
        return rest.strip_suffix(')').map(|s| s.to_string());
    }
    if let Some(rest) = name.strip_prefix("loc(") {
        return rest.strip_suffix(')').map(|s| s.to_string());
    }
    None
}

fn extract_visited(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("Atom visited(") {
        return rest.strip_suffix(')').map(|s| s.to_string());
    }
    if let Some(rest) = name.strip_prefix("visited(") {
        return rest.strip_suffix(')').map(|s| s.to_string());
    }
    None
}

fn compute_averages(cost: &[Vec<f64>], customers: &[usize], depot_idx: usize) -> (f64, f64) {
    let mut ret_sum = 0.0;
    let mut ret_cnt = 0.0;
    let mut start_sum = 0.0;
    let mut start_cnt = 0.0;

    for &c in customers {
        if let Some(&v) = cost.get(c).and_then(|row| row.get(depot_idx)) {
            if v.is_finite() {
                ret_sum += v;
                ret_cnt += 1.0;
            }
        }
        if let Some(&v) = cost.get(depot_idx).and_then(|row| row.get(c)) {
            if v.is_finite() {
                start_sum += v;
                start_cnt += 1.0;
            }
        }
    }

    (
        if ret_cnt > 0.0 { ret_sum / ret_cnt } else { 0.0 },
        if start_cnt > 0.0 { start_sum / start_cnt } else { 0.0 },
    )
}

struct LlmHeuristic {
    depot_idx: usize,
    n_nodes: usize,
    customer_indices: Vec<usize>,
    loc_facts: Vec<LocFact>,
    visited_facts: Vec<VisitedFact>,
    cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    demand: Vec<f64>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    avg_return_cost: f64,
    avg_start_cost: f64,
    extra_route_penalty: f64,
    unserved_penalty: f64,
    scratch: RefCell<ScratchBuffers>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut nodes: Vec<String> = Vec::new();
        let mut node_index: HashMap<String, usize> = HashMap::new();
        let mut customer_order: Vec<String> = Vec::new();
        let mut customer_set: HashSet<String> = HashSet::new();
        let mut visited_map: HashMap<String, FactRef> = HashMap::new();
        let mut loc_facts_raw: Vec<(String, usize, FactRef)> = Vec::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let name = task
                    .get_fact_name(&ExplicitFact::propositional(var_id, value))
                    .to_string();

                if let Some(inner) = extract_visited(&name) {
                    let node = intern(&inner, &mut nodes, &mut node_index);
                    if !customer_set.contains(inner.as_str()) {
                        customer_set.insert(inner.clone());
                        customer_order.push(inner.clone());
                    }
                    visited_map.entry(inner).or_insert(FactRef { var_id, value });
                } else if let Some(inner) = extract_loc(&name) {
                    let node = intern(&inner, &mut nodes, &mut node_index);
                    loc_facts_raw.push((inner, node, FactRef { var_id, value }));
                }
            }
        }

        let depot_idx = loc_facts_raw
            .iter()
            .find(|(name, _, _)| !customer_set.contains(name.as_str()))
            .map(|(_, node, _)| *node)
            .or_else(|| loc_facts_raw.first().map(|(_, node, _)| *node))
            .unwrap_or(0);

        let ops = task.get_operators();

        for op in &ops {
            let parts: Vec<&str> = op.name().split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "visit-via-depot" {
                intern(parts[1], &mut nodes, &mut node_index);
                intern(parts[2], &mut nodes, &mut node_index);
            } else if parts.len() == 3 && parts[0] == "visit" {
                intern(parts[1], &mut nodes, &mut node_index);
                intern(parts[2], &mut nodes, &mut node_index);
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                intern(parts[1], &mut nodes, &mut node_index);
            }
        }

        let n_nodes = nodes.len();
        let mut cost = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
        for i in 0..n_nodes {
            cost[i][i] = 0.0;
        }
        let mut via_depot_cost = vec![vec![f64::INFINITY; n_nodes]; n_nodes];

        for op in &ops {
            let op_cost = metric_operator_cost_from_initial_values(task, op);
            let parts: Vec<&str> = op.name().split_whitespace().collect();

            if parts.len() == 3 && parts[0] == "visit-via-depot" {
                if let (Some(&a), Some(&b)) = (node_index.get(parts[1]), node_index.get(parts[2])) {
                    via_depot_cost[a][b] = via_depot_cost[a][b].min(op_cost);
                }
            } else if parts.len() == 3 && parts[0] == "visit" {
                if let (Some(&a), Some(&b)) = (node_index.get(parts[1]), node_index.get(parts[2])) {
                    cost[a][b] = cost[a][b].min(op_cost);
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&a) = node_index.get(parts[1]) {
                    cost[a][depot_idx] = cost[a][depot_idx].min(op_cost);
                }
            }
        }

        let mut customer_indices: Vec<usize> = Vec::new();
        for name in &customer_order {
            if let Some(&node) = node_index.get(name.as_str()) {
                customer_indices.push(node);
            }
        }

        let mut demand = vec![0.0; n_nodes];
        let mut capacity: Option<f64> = None;
        let mut max_vehicles: Option<f64> = None;
        let mut load_var: Option<usize> = None;
        let mut vehicles_var: Option<usize> = None;
        let initial_numeric = task.get_initial_numeric_state_values();

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name();
            if name.ends_with("load()") {
                load_var = Some(id);
            } else if name.ends_with("max_vehicles()") {
                if let Some(&v) = initial_numeric.get(id) {
                    if v.is_finite() && v > 0.0 {
                        max_vehicles = Some(v);
                    }
                }
            } else if name.ends_with("vehicles()") {
                vehicles_var = Some(id);
            } else if name.ends_with("capacity()") {
                if let Some(&v) = initial_numeric.get(id) {
                    if v.is_finite() && v > 0.0 {
                        capacity = Some(v);
                    }
                }
            }
        }

        for cname in &customer_order {
            let suffix = format!("demand({})", cname);
            for (id, var) in task.numeric_variables().iter().enumerate() {
                if var.name().ends_with(&suffix) {
                    if let (Some(&v), Some(&node)) =
                        (initial_numeric.get(id), node_index.get(cname.as_str()))
                    {
                        if v.is_finite() && v >= 0.0 {
                            demand[node] = v;
                        }
                    }
                    break;
                }
            }
        }

        if let Some(load_var) = load_var {
            for op in &ops {
                let parts: Vec<&str> = op.name().split_whitespace().collect();
                let to_name = if parts.len() == 3
                    && (parts[0] == "visit" || parts[0] == "visit-via-depot")
                {
                    Some(parts[2])
                } else {
                    None
                };

                if let Some(to_name) = to_name {
                    if let Some(&to_idx) = node_index.get(to_name) {
                        let rhs = op
                            .assignment_effects()
                            .iter()
                            .find(|a| a.affected_var_id() == load_var)
                            .and_then(|a| initial_numeric.get(a.var_id()).copied())
                            .unwrap_or(0.0);

                        if demand[to_idx] == 0.0 && rhs.is_finite() && rhs > 0.0 {
                            demand[to_idx] = rhs;
                        }
                    }
                }
            }
        }

        let (avg_return_cost, avg_start_cost) =
            compute_averages(&cost, &customer_indices, depot_idx);

        let extra_route_penalty = if avg_return_cost + avg_start_cost > 0.0 {
            (avg_return_cost + avg_start_cost) * 2.0
        } else {
            1.0
        };
        let unserved_penalty = extra_route_penalty
            .max(avg_return_cost + avg_start_cost)
            .max(1.0);

        let mut visited_facts: Vec<VisitedFact> = Vec::with_capacity(customer_order.len());
        for name in &customer_order {
            if let (Some(&node), Some(&fact)) =
                (node_index.get(name.as_str()), visited_map.get(name.as_str()))
            {
                visited_facts.push(VisitedFact { node, fact });
            }
        }

        let loc_facts: Vec<LocFact> = loc_facts_raw
            .into_iter()
            .map(|(_, node, fact)| LocFact { node, fact })
            .collect();

        Ok(Self {
            depot_idx,
            n_nodes,
            customer_indices,
            loc_facts,
            visited_facts,
            cost,
            via_depot_cost,
            demand,
            capacity,
            max_vehicles,
            load_var,
            vehicles_var,
            avg_return_cost,
            avg_start_cost,
            extra_route_penalty,
            unserved_penalty,
            scratch: RefCell::new(ScratchBuffers {
                numeric: vec![0.0; task.numeric_variables().len()],
                unvisited: Vec::new(),
            }),
        })
    }

    fn direct_cost(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }
        if let Some(&c) = self.cost.get(from).and_then(|row| row.get(to)) {
            if c.is_finite() {
                return c;
            }
        }
        if to == self.depot_idx {
            return self.avg_return_cost;
        }
        if from == self.depot_idx {
            return self.avg_start_cost;
        }
        f64::INFINITY
    }

    fn start_cost(&self, to: usize) -> f64 {
        if to == self.depot_idx {
            return 0.0;
        }
        if let Some(&c) = self.cost.get(self.depot_idx).and_then(|row| row.get(to)) {
            if c.is_finite() {
                return c;
            }
        }
        self.avg_start_cost
    }

    fn return_cost(&self, from: usize) -> f64 {
        if from == self.depot_idx {
            return 0.0;
        }
        if let Some(&c) = self.cost.get(from).and_then(|row| row.get(self.depot_idx)) {
            if c.is_finite() {
                return c;
            }
        }
        self.avg_return_cost
    }

    fn new_route_cost(&self, from: usize, to: usize) -> f64 {
        if from == self.depot_idx {
            return self.start_cost(to);
        }

        let via = self
            .via_depot_cost
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(f64::INFINITY);

        let r = self.return_cost(from);
        let s = self.start_cost(to);
        let combined = if r.is_finite() && s.is_finite() {
            r + s
        } else {
            f64::INFINITY
        };

        if via.is_finite() && (via < combined || !combined.is_finite()) {
            via
        } else {
            combined
        }
    }

    fn can_fit(&self, load: f64, node: usize) -> bool {
        match self.capacity {
            Some(cap) if cap.is_finite() => {
                let d = self.demand.get(node).copied().unwrap_or(0.0);
                load + d <= cap + 1e-9
            }
            _ => true,
        }
    }

    fn best_from(
        &self,
        from: usize,
        load: f64,
        unvisited: &[usize],
        feasible_only: bool,
    ) -> Option<(usize, f64)> {
        let mut best_pos: Option<usize> = None;
        let mut best_cost = f64::INFINITY;

        for (pos, &to) in unvisited.iter().enumerate() {
            if feasible_only && !self.can_fit(load, to) {
                continue;
            }
            let c = self.direct_cost(from, to);
            if c < best_cost {
                best_cost = c;
                best_pos = Some(pos);
            }
        }

        best_pos.map(|pos| (pos, best_cost))
    }

    fn best_new_route_from(
        &self,
        from: usize,
        unvisited: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_pos: Option<usize> = None;
        let mut best_cost = f64::INFINITY;

        for (pos, &to) in unvisited.iter().enumerate() {
            if !self.can_fit(0.0, to) {
                continue;
            }
            let c = self.new_route_cost(from, to);
            if c < best_cost {
                best_cost = c;
                best_pos = Some(pos);
            }
        }

        best_pos.map(|pos| (pos, best_cost))
    }

    fn consume_new_vehicle(&self, vehicles_left: &mut f64, total: &mut f64) {
        if let Some(max) = self.max_vehicles {
            if max.is_finite() {
                if *vehicles_left >= 1.0 {
                    *vehicles_left -= 1.0;
                } else {
                    *total += self.extra_route_penalty;
                }
            }
        }
    }

    fn greedy_estimate(
        &self,
        start: usize,
        start_load: f64,
        vehicles_used_raw: f64,
        unvisited: &mut Vec<usize>,
    ) -> f64 {
        if unvisited.is_empty() {
            return self.return_cost(start).max(0.0);
        }

        let vehicles_used = if vehicles_used_raw.is_finite() && vehicles_used_raw >= 0.0 {
            vehicles_used_raw
        } else {
            0.0
        };

        let mut current = start;
        let mut load = if start_load.is_finite() && start_load >= 0.0 {
            start_load
        } else {
            0.0
        };
        let mut total = 0.0;

        let mut vehicles_left = match self.max_vehicles {
            Some(max) if max.is_finite() => (max - vehicles_used).max(0.0),
            _ => f64::INFINITY,
        };

        while !unvisited.is_empty() {
            if let Some((pos, c)) = self.best_from(current, load, unvisited, true) {
                let node = unvisited.swap_remove(pos);
                total += c;
                load += self.demand[node];
                current = node;
                continue;
            }

            if current != self.depot_idx {
                self.consume_new_vehicle(&mut vehicles_left, &mut total);

                if let Some((pos, c)) = self.best_new_route_from(current, unvisited) {
                    let node = unvisited.swap_remove(pos);
                    total += c;
                    current = node;
                    load = self.demand[node];
                    continue;
                }

                total += self.return_cost(current);
                current = self.depot_idx;
                load = 0.0;

                if let Some((pos, c)) = self.best_from(current, 0.0, unvisited, false) {
                    let node = unvisited.swap_remove(pos);
                    total += c;
                    current = node;
                    load = self.demand[node];
                    continue;
                } else {
                    break;
                }
            } else {
                if let Some((pos, c)) = self.best_from(current, load, unvisited, false) {
                    let node = unvisited.swap_remove(pos);
                    total += c;
                    current = node;
                    load = self.demand[node];
                    continue;
                } else {
                    break;
                }
            }
        }

        if current != self.depot_idx {
            total += self.return_cost(current);
        }

        if !unvisited.is_empty() {
            total += self.unserved_penalty * unvisited.len() as f64;
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

        let mut current = self.depot_idx;
        let mut found_loc = false;
        for loc in &self.loc_facts {
            let fact = ExplicitFact::propositional(loc.fact.var_id, loc.fact.value);
            if fact.is_hold(registry.view(state)) {
                current = loc.node;
                found_loc = true;
                break;
            }
        }
        if !found_loc {
            current = self.depot_idx;
        }

        let mut unvisited = {
            let mut scratch = self.scratch.borrow_mut();
            std::mem::take(&mut scratch.unvisited)
        };
        unvisited.clear();

        for v in &self.visited_facts {
            let fact = ExplicitFact::propositional(v.fact.var_id, v.fact.value);
            if !fact.is_hold(registry.view(state)) {
                unvisited.push(v.node);
            }
        }

        let load_var = self.load_var;
        let vehicles_var = self.vehicles_var;
        let numeric_len = task.numeric_variables().len();

        let (load, vehicles_used) = {
            let mut scratch = self.scratch.borrow_mut();
            if scratch.numeric.len() != numeric_len {
                scratch.numeric.resize(numeric_len, 0.0);
            }

            registry
                .view(state)
                .fill_numeric(&mut scratch.numeric)
                .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

            let load = match load_var.and_then(|id| scratch.numeric.get(id).copied()) {
                Some(v) if v.is_finite() && v >= 0.0 => v,
                _ => 0.0,
            };
            let vehicles_used = match vehicles_var
                .and_then(|id| scratch.numeric.get(id).copied())
            {
                Some(v) if v.is_finite() && v >= 0.0 => v,
                _ => 0.0,
            };

            (load, vehicles_used)
        };

        let h = self.greedy_estimate(current, load, vehicles_used, &mut unvisited);

        {
            let mut scratch = self.scratch.borrow_mut();
            scratch.unvisited = unvisited;
        }

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
