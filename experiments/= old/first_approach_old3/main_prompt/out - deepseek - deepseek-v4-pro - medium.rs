use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

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


#[derive(Clone, Copy, PartialEq, Eq)]
enum OpKind { Visit, Via, Return }

struct RawOp {
    kind: OpKind,
    from: String,
    to: String,
    cost: f64,
}

fn parse_pred_arg(name: &str, pred: &str) -> Option<String> {
    let needle = format!("{pred}(");
    let pos = name.find(&needle)?;
    let before = &name[..pos];
    if before.trim_end().ends_with("not") {
        return None;
    }
    let after = &name[pos + needle.len()..];
    let end = after.find(')')?;
    Some(after[..end].to_string())
}

fn parse_demand_arg(name: &str) -> Option<String> {
    let needle = "demand(";
    let pos = name.find(needle)?;
    let after = &name[pos + needle.len()..];
    let end = after.find(')')?;
    Some(after[..end].to_string())
}

fn parse_op_name(name: &str) -> Option<(&str, &str, &str)> {
    let mut parts = name.split_whitespace();
    let action = parts.next()?;
    match action {
        "visit" => {
            let from = parts.next()?;
            let to = parts.next()?;
            Some((action, from, to))
        }
        "visit-via-depot" => {
            let from = parts.next()?;
            let to = parts.next()?;
            Some((action, from, to))
        }
        "return-to-depot" => {
            let from = parts.next()?;
            Some((action, from, ""))
        }
        _ => None,
    }
}

fn sorted_customer_indices(customers: &[usize], costs: &[f64]) -> Vec<usize> {
    let mut v = customers.to_vec();
    v.sort_by(|a, b| {
        let ca = if costs[*a].is_finite() { costs[*a] } else { f64::INFINITY };
        let cb = if costs[*b].is_finite() { costs[*b] } else { f64::INFINITY };
        ca.partial_cmp(&cb)
            .unwrap_or(Ordering::Equal)
            .then_with(|| (*a).cmp(*b))
    });
    v
}


struct LlmHeuristic {
    n_places: usize,
    n_numeric: usize,
    n_prop: usize,
    depot_idx: usize,
    customer_indices: Vec<usize>,
    direct_cost: Vec<Vec<f64>>,
    via_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,
    direct_sorted: Vec<Vec<usize>>,
    via_sorted: Vec<Vec<usize>>,
    loc_facts: Vec<(usize, usize, usize)>,
    visited_facts: Vec<(usize, usize, usize)>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    max_vehicles_var: Option<usize>,
    capacity: f64,
    demands: Vec<f64>,
    has_capacity_info: bool,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut place_set = BTreeSet::new();
        let mut raw_ops = Vec::new();

        for op in task.get_operators() {
            if let Some((action, from, to)) = parse_op_name(op.name()) {
                let cost = metric_operator_cost_from_initial_values(task, op);
                match action {
                    "visit" => {
                        place_set.insert(from.to_string());
                        place_set.insert(to.to_string());
                        raw_ops.push(RawOp {
                            kind: OpKind::Visit,
                            from: from.to_string(),
                            to: to.to_string(),
                            cost,
                        });
                    }
                    "visit-via-depot" => {
                        place_set.insert(from.to_string());
                        place_set.insert(to.to_string());
                        raw_ops.push(RawOp {
                            kind: OpKind::Via,
                            from: from.to_string(),
                            to: to.to_string(),
                            cost,
                        });
                    }
                    "return-to-depot" => {
                        place_set.insert(from.to_string());
                        raw_ops.push(RawOp {
                            kind: OpKind::Return,
                            from: from.to_string(),
                            to: String::new(),
                            cost,
                        });
                    }
                    _ => {}
                }
            }
        }

        if place_set.is_empty() {
            return Err("CVRP heuristic: no grounded operators found".to_string().into());
        }

        let visit_from: BTreeSet<String> = raw_ops
            .iter()
            .filter(|o| o.kind == OpKind::Visit)
            .map(|o| o.from.clone())
            .collect();
        let visit_to: BTreeSet<String> = raw_ops
            .iter()
            .filter(|o| o.kind == OpKind::Visit)
            .map(|o| o.to.clone())
            .collect();

        let diff: Vec<String> = visit_from.difference(&visit_to).cloned().collect();
        let depot_name = diff
            .iter()
            .find(|p| p.as_str() == "d1")
            .cloned()
            .or_else(|| diff.first().cloned())
            .ok_or_else(|| "CVRP heuristic: could not identify depot".to_string())?;

        let places: Vec<String> = place_set.into_iter().collect();
        let mut place_index = HashMap::new();
        for (i, p) in places.iter().enumerate() {
            place_index.insert(p.clone(), i);
        }

        let n = places.len();
        let depot_idx = *place_index
            .get(&depot_name)
            .ok_or_else(|| "CVRP heuristic: depot missing from places".to_string())?;
        let customer_indices: Vec<usize> = places
            .iter()
            .enumerate()
            .filter(|(_, p)| p.as_str() != depot_name.as_str())
            .map(|(i, _)| i)
            .collect();

        let mut direct = vec![vec![f64::INFINITY; n]; n];
        let mut via = vec![vec![f64::INFINITY; n]; n];
        let mut return_cost = vec![f64::INFINITY; n];

        for op in &raw_ops {
            let a = place_index.get(&op.from);
            let b = place_index.get(&op.to);
            match op.kind {
                OpKind::Visit => {
                    if let (Some(&a), Some(&b)) = (a, b) {
                        direct[a][b] = direct[a][b].min(op.cost);
                    }
                }
                OpKind::Via => {
                    if let (Some(&a), Some(&b)) = (a, b) {
                        via[a][b] = via[a][b].min(op.cost);
                    }
                }
                OpKind::Return => {
                    if let Some(&a) = a {
                        return_cost[a] = return_cost[a].min(op.cost);
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                if direct[i][j].is_infinite() && direct[j][i].is_finite() {
                    direct[i][j] = direct[j][i];
                }
                if via[i][j].is_infinite() && via[j][i].is_finite() {
                    via[i][j] = via[j][i];
                }
            }
        }

        for &i in &customer_indices {
            if return_cost[i].is_infinite() && direct[depot_idx][i].is_finite() {
                return_cost[i] = direct[depot_idx][i];
            }
            if direct[depot_idx][i].is_infinite() && return_cost[i].is_finite() {
                direct[depot_idx][i] = return_cost[i];
            }
        }

        for &i in &customer_indices {
            for &j in &customer_indices {
                if i != j
                    && via[i][j].is_infinite()
                    && return_cost[i].is_finite()
                    && direct[depot_idx][j].is_finite()
                {
                    via[i][j] = return_cost[i] + direct[depot_idx][j];
                }
            }
        }

        let direct_sorted: Vec<Vec<usize>> = (0..n)
            .map(|i| sorted_customer_indices(&customer_indices, &direct[i]))
            .collect();
        let via_sorted: Vec<Vec<usize>> = (0..n)
            .map(|i| sorted_customer_indices(&customer_indices, &via[i]))
            .collect();

        let mut loc_facts = Vec::new();
        let mut visited_facts = Vec::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(obj) = parse_pred_arg(&fact_name, "loc(") {
                    if let Some(&idx) = place_index.get(&obj) {
                        loc_facts.push((var_id, value, idx));
                    }
                }
                if let Some(obj) = parse_pred_arg(&fact_name, "visited(") {
                    if let Some(&idx) = place_index.get(&obj) {
                        visited_facts.push((var_id, value, idx));
                    }
                }
            }
        }

        let mut load_var = None;
        let mut vehicles_var = None;
        let mut max_vehicles_var = None;
        let mut capacity_var = None;
        let mut demand_pairs: Vec<(usize, usize)> = Vec::new();

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name();
            if name.ends_with("load()") {
                load_var = Some(id);
            } else if name.ends_with("max_vehicles()") {
                max_vehicles_var = Some(id);
            } else if name.ends_with("vehicles()") {
                vehicles_var = Some(id);
            } else if name.ends_with("capacity()") {
                capacity_var = Some(id);
            }

            if let Some(cust) = parse_demand_arg(name) {
                if let Some(&idx) = place_index.get(&cust) {
                    demand_pairs.push((idx, id));
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let has_capacity_info = load_var.is_some()
            && capacity_var.is_some()
            && demand_pairs.len() == customer_indices.len();
        let capacity = if has_capacity_info {
            initial_numeric[capacity_var.unwrap()]
        } else {
            f64::INFINITY
        };

        let mut demands = vec![0.0; n];
        if has_capacity_info {
            for (idx, var_id) in &demand_pairs {
                demands[*idx] = initial_numeric[*var_id];
            }
        }

        Ok(Self {
            n_places: n,
            n_numeric: task.numeric_variables().len(),
            n_prop: task.variables().len(),
            depot_idx,
            customer_indices,
            direct_cost: direct,
            via_cost: via,
            return_cost,
            direct_sorted,
            via_sorted,
            loc_facts,
            visited_facts,
            load_var,
            vehicles_var,
            max_vehicles_var,
            capacity,
            demands,
            has_capacity_info,
        })
    }

    fn best_direct(
        &self,
        i: usize,
        unvisited: &[bool],
        load: f64,
        cap: f64,
    ) -> Option<(usize, f64)> {
        for &j in &self.direct_sorted[i] {
            if unvisited[j] {
                let cost = self.direct_cost[i][j];
                if !cost.is_finite() {
                    continue;
                }
                if self.has_capacity_info && load + self.demands[j] > cap + 1e-9 {
                    continue;
                }
                return Some((j, cost));
            }
        }
        None
    }

    fn nearest_via(&self, i: usize, unvisited: &[bool]) -> Option<(usize, f64)> {
        for &j in &self.via_sorted[i] {
            if unvisited[j] {
                let cost = self.via_cost[i][j];
                if cost.is_finite() {
                    return Some((j, cost));
                }
            }
        }
        None
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

        let mut numeric_values = vec![0.0; self.n_numeric];
        registry
            .view(state)
            .fill_numeric(&mut numeric_values)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

        let mut propositional_values = vec![0usize; self.n_prop];
        state.fill_state(registry, &mut propositional_values);

        let mut cur = self.depot_idx;
        for &(var_id, value, idx) in &self.loc_facts {
            if propositional_values[var_id] == value {
                cur = idx;
                break;
            }
        }

        let mut unvisited = vec![true; self.n_places];
        for &(var_id, value, idx) in &self.visited_facts {
            if propositional_values[var_id] == value {
                unvisited[idx] = false;
            }
        }
        unvisited[self.depot_idx] = false;

        let mut remaining = 0usize;
        for &b in &unvisited {
            if b {
                remaining += 1;
            }
        }

        let mut load = if self.has_capacity_info {
            numeric_values[self.load_var.unwrap()]
        } else {
            0.0
        };
        let mut used_vehicles = self
            .vehicles_var
            .map(|id| numeric_values[id])
            .unwrap_or(1.0);
        let max_vehicles = self
            .max_vehicles_var
            .map(|id| numeric_values[id])
            .unwrap_or(f64::INFINITY);
        let cap = self.capacity;

        let mut h = 0.0;

        while remaining > 0 {
            if cur == self.depot_idx {
                if let Some((next, cost)) =
                    self.best_direct(self.depot_idx, &unvisited, load, cap)
                {
                    h += cost;
                    if self.has_capacity_info {
                        load += self.demands[next];
                    }
                    cur = next;
                    unvisited[next] = false;
                    remaining -= 1;
                } else {
                    break;
                }
            } else {
                let direct_candidate =
                    self.best_direct(cur, &unvisited, load, cap);
                let via_candidate = if used_vehicles + 1.0 <= max_vehicles + 1e-9 {
                    self.nearest_via(cur, &unvisited)
                } else {
                    None
                };

                let chosen = match (direct_candidate, via_candidate) {
                    (Some((jd, cd)), Some((jv, cv))) => {
                        if cv < cd - 1e-9 {
                            Some((jv, cv, true))
                        } else {
                            Some((jd, cd, false))
                        }
                    }
                    (Some((jd, cd)), None) => Some((jd, cd, false)),
                    (None, Some((jv, cv))) => Some((jv, cv, true)),
                    (None, None) => None,
                };

                if let Some((next, cost, via_mode)) = chosen {
                    h += cost;
                    if self.has_capacity_info {
                        if via_mode {
                            load = self.demands[next];
                        } else {
                            load += self.demands[next];
                        }
                    }
                    if via_mode {
                        used_vehicles += 1.0;
                    }
                    cur = next;
                    unvisited[next] = false;
                    remaining -= 1;
                } else {
                    break;
                }
            }
        }

        if remaining == 0 && cur != self.depot_idx {
            let rc = self.return_cost[cur];
            if rc.is_finite() {
                h += rc;
            } else if self.direct_cost[self.depot_idx][cur].is_finite() {
                h += self.direct_cost[self.depot_idx][cur];
            }
        }

        Ok(h.max(0.0))
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
