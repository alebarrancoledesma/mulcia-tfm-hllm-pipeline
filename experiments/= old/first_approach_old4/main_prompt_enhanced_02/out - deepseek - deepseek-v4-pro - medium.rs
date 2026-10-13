use std::cell::RefCell;
use std::cmp::Ordering;
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

const LARGE_PENALTY: f64 = 1.0e12;

#[derive(Clone, Copy, Debug)]
struct FactId {
    var_id: usize,
    true_value: usize,
}

struct LocFact {
    fact: FactId,
    object: String,
}

#[inline(always)]
fn fact_holds(props: &[usize], fact: &FactId) -> bool {
    props[fact.var_id] == fact.true_value
}

fn extract_object(text: &str) -> Option<String> {
    let start = text.find('(')?;
    let end = text.rfind(')')?;
    if start < end {
        Some(text[start + 1..end].to_string())
    } else {
        None
    }
}

fn parse_visit_action(name: &str) -> Option<(String, String)> {
    if let Some(rest) = name.strip_prefix("visit-via-depot ") {
        let mut parts = rest.split_whitespace();
        let from = parts.next()?.to_string();
        let to = parts.next()?.to_string();
        if parts.next().is_some() {
            return None;
        }
        Some((from, to))
    } else if let Some(rest) = name.strip_prefix("visit ") {
        let mut parts = rest.split_whitespace();
        let from = parts.next()?.to_string();
        let to = parts.next()?.to_string();
        if parts.next().is_some() {
            return None;
        }
        Some((from, to))
    } else {
        None
    }
}

fn parse_return_action(name: &str) -> Option<String> {
    let rest = name.strip_prefix("return-to-depot ")?;
    Some(rest.split_whitespace().next()?.to_string())
}

fn parse_function_arg(name: &str, marker: &str) -> Option<String> {
    let pos = name.find(marker)?;
    let after = name[pos + marker.len()..].trim_start();

    if after.starts_with('(') {
        let inner = &after[1..];
        let close = inner.find(')');
        let comma = inner.find(',');
        let space = inner.find(' ');
        let end = close
            .unwrap_or(inner.len())
            .min(comma.unwrap_or(inner.len()))
            .min(space.unwrap_or(inner.len()));
        Some(inner[..end].to_string())
    } else {
        let close = after.find(')');
        let comma = after.find(',');
        let space = after.find(' ');
        let end = close
            .unwrap_or(after.len())
            .min(comma.unwrap_or(after.len()))
            .min(space.unwrap_or(after.len()));
        Some(after[..end].to_string())
    }
}

fn parse_two_args(name: &str, marker: &str) -> Option<(String, String)> {
    let pos = name.find(marker)?;
    let after = name[pos + marker.len()..].trim_start();

    if after.starts_with('(') {
        let inner = &after[1..];
        let end = inner.find(')').unwrap_or(inner.len());
        let inner = &inner[..end];
        let parts: Vec<&str> = inner
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|p| !p.is_empty())
            .collect();
        if parts.len() == 2 {
            Some((parts[0].to_string(), parts[1].to_string()))
        } else {
            None
        }
    } else {
        let parts: Vec<&str> = after
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|p| !p.is_empty())
            .collect();
        if parts.len() >= 2 {
            Some((parts[0].to_string(), parts[1].to_string()))
        } else {
            None
        }
    }
}

fn find_numeric_id<F>(task: &dyn AbstractNumericTask, mut pred: F) -> Option<usize>
where
    F: FnMut(&str) -> bool,
{
    for (id, var) in task.numeric_variables().iter().enumerate() {
        let name = var.name().to_string();
        if pred(&name) {
            return Some(id);
        }
    }
    None
}

struct LlmHeuristic {
    customer_count: usize,
    depot_idx: usize,
    loc_facts: Vec<LocFact>,
    visited_facts: Vec<FactId>,
    obj_to_idx: HashMap<String, usize>,
    load_id: usize,
    vehicles_id: usize,
    numeric_var_count: usize,
    prop_var_count: usize,
    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    depot_to_cust: Vec<f64>,
    cust_to_depot: Vec<f64>,
    cust_cust: Vec<Vec<f64>>,
    via_cust: Vec<Vec<f64>>,
    nearest_depot: Vec<usize>,
    nearest_direct: Vec<Vec<usize>>,
    nearest_via: Vec<Vec<usize>>,
    avg_roundtrip: f64,
    numeric_scratch: RefCell<Vec<f64>>,
    prop_scratch: RefCell<Vec<usize>>,
    unvisited_scratch: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let prop_count = task.variables().len();
        let numeric_count = task.numeric_variables().len();
        let initial_numeric = task.get_initial_numeric_state_values();

        let mut loc_facts: Vec<LocFact> = Vec::new();
        let mut visited_facts_temp: Vec<(FactId, String)> = Vec::new();
        let mut loc_objects: HashSet<String> = HashSet::new();
        let mut visited_objects: HashSet<String> = HashSet::new();

        for var_id in 0..prop_count {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = format!("{}", task.get_fact_name(&fact));

                if name.starts_with("Atom loc(") {
                    if let Some(obj) = extract_object(&name) {
                        loc_objects.insert(obj.clone());
                        loc_facts.push(LocFact {
                            fact: FactId {
                                var_id,
                                true_value: value,
                            },
                            object: obj,
                        });
                    }
                } else if name.starts_with("Atom visited(") {
                    if let Some(obj) = extract_object(&name) {
                        visited_objects.insert(obj.clone());
                        visited_facts_temp.push((
                            FactId {
                                var_id,
                                true_value: value,
                            },
                            obj,
                        ));
                    }
                }
            }
        }

        let mut customer_objects: Vec<String> = visited_objects.into_iter().collect();
        customer_objects.sort();
        let customer_count = customer_objects.len();
        let depot_idx = customer_count;

        let mut obj_to_idx: HashMap<String, usize> = HashMap::with_capacity(customer_count + 1);
        for (i, obj) in customer_objects.iter().cloned().enumerate() {
            obj_to_idx.insert(obj, i);
        }

        let mut depot_object: Option<String> = None;
        for obj in loc_objects {
            if !obj_to_idx.contains_key(&obj) {
                depot_object = Some(obj);
                break;
            }
        }

        let depot_object = depot_object.ok_or_else(|| -> HeuristicBuildError {
            "Could not identify depot from loc/visited facts".to_string().into()
        })?;
        obj_to_idx.insert(depot_object.clone(), depot_idx);

        let mut visited_facts = vec![
            FactId {
                var_id: 0,
                true_value: 0,
            };
            customer_count
        ];
        for (fact, obj) in visited_facts_temp {
            if let Some(&idx) = obj_to_idx.get(&obj) {
                if idx < customer_count {
                    visited_facts[idx] = fact;
                }
            }
        }

        let load_id = find_numeric_id(task, |n| n.contains("load()") && !n.contains("total"))
            .ok_or_else(|| -> HeuristicBuildError {
                "missing load numeric variable".to_string().into()
            })?;
        let vehicles_id =
            find_numeric_id(task, |n| n.contains("vehicles()") && !n.contains("max"))
                .ok_or_else(|| -> HeuristicBuildError {
                    "missing vehicles numeric variable".to_string().into()
                })?;

        let capacity_id = find_numeric_id(task, |n| n.contains("capacity") && !n.contains("max_vehicles"));
        let capacity = capacity_id
            .map(|id| initial_numeric[id])
            .unwrap_or(1.0e18)
            .max(1.0);

        let max_vehicles_id = find_numeric_id(task, |n| n.contains("max_vehicles"));
        let max_vehicles = max_vehicles_id
            .map(|id| initial_numeric[id])
            .unwrap_or((customer_count + 1) as f64)
            .max(1.0);

        let mut demands = vec![0.0; customer_count];
        let mut found_demand = false;
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if name.contains("demand(") {
                if let Some(cust) = parse_function_arg(&name, "demand") {
                    if let Some(&idx) = obj_to_idx.get(&cust) {
                        if idx < customer_count {
                            demands[idx] = initial_numeric[id];
                            found_demand = true;
                        }
                    }
                }
            }
        }
        if !found_demand {
            for d in demands.iter_mut() {
                *d = 1.0;
            }
        }

        let inf = f64::INFINITY;
        let mut depot_to_cust = vec![inf; customer_count];
        let mut cust_to_depot = vec![inf; customer_count];
        let mut cust_cust = vec![vec![inf; customer_count]; customer_count];
        let mut via_cust = vec![vec![inf; customer_count]; customer_count];

        for op in task.get_operators().iter() {
            let name = op.name().to_string();
            let cost = metric_operator_cost_from_initial_values(task, op);

            if let Some((from, to)) = parse_visit_action(&name) {
                let from_idx = obj_to_idx.get(&from).copied();
                let to_idx = obj_to_idx.get(&to).copied();

                if name.starts_with("visit-via-depot ") {
                    if let (Some(f), Some(t)) = (from_idx, to_idx) {
                        if f < customer_count && t < customer_count && cost < via_cust[f][t] {
                            via_cust[f][t] = cost;
                        }
                    }
                } else if name.starts_with("visit ") {
                    if let (Some(f), Some(t)) = (from_idx, to_idx) {
                        if f == depot_idx && t < customer_count {
                            if cost < depot_to_cust[t] {
                                depot_to_cust[t] = cost;
                            }
                        } else if f < customer_count && t < customer_count {
                            if cost < cust_cust[f][t] {
                                cust_cust[f][t] = cost;
                            }
                        }
                    }
                }
            } else if let Some(from) = parse_return_action(&name) {
                if let Some(f) = obj_to_idx.get(&from).copied() {
                    if f < customer_count && cost < cust_to_depot[f] {
                        cust_to_depot[f] = cost;
                    }
                }
            }
        }

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if name.contains("travel-cost") && !name.contains("via-depot") {
                if let Some((a, b)) = parse_two_args(&name, "travel-cost") {
                    let val = initial_numeric[id];
                    if a == depot_object {
                        if let Some(&t) = obj_to_idx.get(&b) {
                            if t < customer_count {
                                depot_to_cust[t] = depot_to_cust[t].min(val);
                            }
                        }
                    } else if b == depot_object {
                        if let Some(&f) = obj_to_idx.get(&a) {
                            if f < customer_count {
                                cust_to_depot[f] = cust_to_depot[f].min(val);
                            }
                        }
                    } else if let (Some(&f), Some(&t)) = (obj_to_idx.get(&a), obj_to_idx.get(&b)) {
                        if f < customer_count && t < customer_count {
                            cust_cust[f][t] = cust_cust[f][t].min(val);
                        }
                    }
                }
            }
        }

        for i in 0..customer_count {
            for j in 0..customer_count {
                if i == j {
                    continue;
                }
                let fallback = cust_to_depot[i] + depot_to_cust[j];
                if via_cust[i][j].is_infinite()
                    && cust_to_depot[i].is_finite()
                    && depot_to_cust[j].is_finite()
                {
                    via_cust[i][j] = fallback;
                }
                if cust_cust[i][j].is_infinite()
                    && cust_to_depot[i].is_finite()
                    && depot_to_cust[j].is_finite()
                {
                    cust_cust[i][j] = fallback;
                }
            }
        }

        if depot_to_cust.iter().any(|x| x.is_infinite())
            || cust_to_depot.iter().any(|x| x.is_infinite())
        {
            return Err(
                "Incomplete directed depot/customer travel costs after operator and PNE fallback"
                    .to_string()
                    .into(),
            );
        }

        let mut nearest_depot: Vec<usize> = (0..customer_count).collect();
        nearest_depot.sort_by(|&a, &b| {
            depot_to_cust[a]
                .partial_cmp(&depot_to_cust[b])
                .unwrap_or(Ordering::Equal)
        });

        let mut nearest_direct: Vec<Vec<usize>> = Vec::with_capacity(customer_count);
        let mut nearest_via: Vec<Vec<usize>> = Vec::with_capacity(customer_count);

        for i in 0..customer_count {
            let mut direct: Vec<usize> = (0..customer_count).filter(|&j| j != i).collect();
            direct.sort_by(|&a, &b| {
                cust_cust[i][a]
                    .partial_cmp(&cust_cust[i][b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_direct.push(direct);

            let mut via: Vec<usize> = (0..customer_count).filter(|&j| j != i).collect();
            via.sort_by(|&a, &b| {
                via_cust[i][a]
                    .partial_cmp(&via_cust[i][b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_via.push(via);
        }

        let avg_depot_to = if customer_count > 0 {
            depot_to_cust.iter().sum::<f64>() / customer_count as f64
        } else {
            0.0
        };
        let avg_cust_to_depot = if customer_count > 0 {
            cust_to_depot.iter().sum::<f64>() / customer_count as f64
        } else {
            0.0
        };
        let avg_roundtrip = avg_depot_to + avg_cust_to_depot;

        Ok(Self {
            customer_count,
            depot_idx,
            loc_facts,
            visited_facts,
            obj_to_idx,
            load_id,
            vehicles_id,
            numeric_var_count: numeric_count,
            prop_var_count: prop_count,
            demands,
            capacity,
            max_vehicles,
            depot_to_cust,
            cust_to_depot,
            cust_cust,
            via_cust,
            nearest_depot,
            nearest_direct,
            nearest_via,
            avg_roundtrip,
            numeric_scratch: RefCell::new(vec![0.0; numeric_count]),
            prop_scratch: RefCell::new(vec![0; prop_count]),
            unvisited_scratch: RefCell::new(vec![false; customer_count]),
        })
    }

    fn choose_next(
        &self,
        cur: usize,
        unvisited: &[bool],
        sim_load: f64,
        sim_vehicles: f64,
    ) -> Option<(usize, f64, u8)> {
        let n = self.customer_count;
        let cap = self.capacity;
        let max_v = self.max_vehicles;
        let eps = 1.0e-9;

        if cur == self.depot_idx {
            for &next in &self.nearest_depot {
                if unvisited[next]
                    && sim_load + self.demands[next] <= cap + eps
                    && self.depot_to_cust[next].is_finite()
                {
                    return Some((next, self.depot_to_cust[next], 1));
                }
            }

            let mut best: Option<(usize, f64, u8)> = None;
            for next in 0..n {
                if unvisited[next]
                    && sim_load + self.demands[next] <= cap + eps
                    && self.depot_to_cust[next].is_finite()
                {
                    let cost = self.depot_to_cust[next];
                    if best.as_ref().map_or(true, |(_, best_cost, _)| cost < *best_cost) {
                        best = Some((next, cost, 1));
                    }
                }
            }
            return best;
        }

        let mut best: Option<(usize, f64, u8)> = None;

        for &next in &self.nearest_direct[cur] {
            if unvisited[next]
                && sim_load + self.demands[next] <= cap + eps
                && self.cust_cust[cur][next].is_finite()
            {
                best = Some((next, self.cust_cust[cur][next], 1));
                break;
            }
        }

        for &next in &self.nearest_via[cur] {
            if unvisited[next]
                && sim_vehicles + 1.0 <= max_v + eps
                && self.via_cust[cur][next].is_finite()
            {
                let cost = self.via_cust[cur][next];
                if best.as_ref().map_or(true, |(_, best_cost, _)| cost < *best_cost) {
                    best = Some((next, cost, 2));
                }
                break;
            }
        }

        if best.is_some() {
            return best;
        }

        let mut fallback_best: Option<(usize, f64, u8)> = None;
        for next in 0..n {
            if !unvisited[next] {
                continue;
            }

            if sim_load + self.demands[next] <= cap + eps && self.cust_cust[cur][next].is_finite() {
                let cost = self.cust_cust[cur][next];
                if fallback_best
                    .as_ref()
                    .map_or(true, |(_, best_cost, _)| cost < *best_cost)
                {
                    fallback_best = Some((next, cost, 1));
                }
            }

            if sim_vehicles + 1.0 <= max_v + eps && self.via_cust[cur][next].is_finite() {
                let cost = self.via_cust[cur][next];
                if fallback_best
                    .as_ref()
                    .map_or(true, |(_, best_cost, _)| cost < *best_cost)
                {
                    fallback_best = Some((next, cost, 2));
                }
            }
        }
        fallback_best
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

        let mut numeric_values = self.numeric_scratch.borrow_mut();
        numeric_values.clear();
        numeric_values.resize(self.numeric_var_count, 0.0);
        registry
            .view(state)
            .fill_numeric(&mut *numeric_values)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{:?}", err)))?;

        let mut prop_values = self.prop_scratch.borrow_mut();
        prop_values.clear();
        prop_values.resize(self.prop_var_count, 0);
        state
            .fill_state(registry, &mut *prop_values)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{:?}", err)))?;

        let mut current_idx: Option<usize> = None;
        for lf in &self.loc_facts {
            if fact_holds(&prop_values, &lf.fact) {
                if let Some(&idx) = self.obj_to_idx.get(&lf.object) {
                    current_idx = Some(idx);
                    break;
                }
            }
        }
        let current_idx = match current_idx {
            Some(idx) => idx,
            None => return Ok(LARGE_PENALTY + self.customer_count as f64),
        };

        let mut unvisited = self.unvisited_scratch.borrow_mut();
        unvisited.clear();
        unvisited.resize(self.customer_count, false);
        let mut unvisited_count = 0usize;

        for i in 0..self.customer_count {
            let visited = fact_holds(&prop_values, &self.visited_facts[i]);
            unvisited[i] = !visited;
            if !visited {
                unvisited_count += 1;
            }
        }

        let load = numeric_values[self.load_id];
        let used_vehicles = numeric_values[self.vehicles_id];

        if unvisited_count == 0 {
            if current_idx == self.depot_idx {
                return Ok(0.0);
            }
            let cost = self
                .cust_to_depot
                .get(current_idx)
                .copied()
                .unwrap_or(LARGE_PENALTY);
            return Ok(cost);
        }

        let remaining_demand: f64 = (0..self.customer_count)
            .filter(|&i| unvisited[i])
            .map(|i| self.demands[i])
            .sum();

        let capacity_left = (self.capacity - load).max(0.0);
        let remaining_vehicle_count = (self.max_vehicles - used_vehicles).max(0.0);
        let total_available = capacity_left + remaining_vehicle_count * self.capacity;

        if remaining_demand > total_available + 1.0e-9 {
            return Ok(LARGE_PENALTY + remaining_demand);
        }

        let mut sim_load = load;
        let mut sim_vehicles = used_vehicles;
        let mut cur = current_idx;
        let mut remaining = unvisited_count;
        let mut h_routing = 0.0;

        while remaining > 0 {
            let next = self.choose_next(cur, &unvisited, sim_load, sim_vehicles);
            if let Some((next_idx, cost, mode)) = next {
                unvisited[next_idx] = false;
                remaining -= 1;

                if cur == self.depot_idx || mode == 1 {
                    h_routing += cost;
                    sim_load += self.demands[next_idx];
                } else {
                    h_routing += cost;
                    sim_load = self.demands[next_idx];
                    sim_vehicles += 1.0;
                }

                cur = next_idx;
            } else {
                return Ok(LARGE_PENALTY + remaining as f64);
            }
        }

        if cur != self.depot_idx {
            h_routing += self.cust_to_depot[cur];
        }

        let extra_routes = ((remaining_demand - capacity_left).max(0.0) / self.capacity).ceil();
        let h_capacity = extra_routes * self.avg_roundtrip * 0.5;

        let h = h_routing + h_capacity;
        if !h.is_finite() {
            return Ok(LARGE_PENALTY + remaining_demand);
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

fn no_nested_heuristics(_: &HeuristicSpec) -> Result<Vec<HeuristicSpec>, String> {
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
