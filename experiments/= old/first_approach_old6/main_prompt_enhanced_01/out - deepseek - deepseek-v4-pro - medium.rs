use std::cell::RefCell;
use std::cmp::Ordering;
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

const INF: f64 = 1e30;
const LARGE_PENALTY: f64 = 1e9;

fn extract_paren_arg(name: &str, prefixes: &[&str]) -> Option<String> {
    let name = name.trim();
    for p in prefixes {
        if name.starts_with(p) && name.ends_with(')') {
            let inner = &name[p.len()..name.len() - 1];
            return Some(inner.trim().to_string());
        }
    }
    None
}

fn extract_travel_cost_arg(name: &str) -> Option<String> {
    let name = name.trim();
    if name.starts_with("PNE travel-cost(") && name.ends_with(')') {
        return Some(name["PNE travel-cost(".len()..name.len() - 1].trim().to_string());
    }
    if name.starts_with("travel-cost(") && name.ends_with(')') {
        return Some(name["travel-cost(".len()..name.len() - 1].trim().to_string());
    }
    None
}

fn find_numeric_id(names: &[String], candidates: &[&str]) -> Option<usize> {
    for (i, name) in names.iter().enumerate() {
        let t = name.trim();
        for c in candidates {
            if t == *c {
                return Some(i);
            }
        }
    }
    None
}

struct LlmHeuristic {
    n_customers: usize,
    depot_idx: usize,
    loc_facts: Vec<(usize, usize, usize)>,
    visited_facts: Vec<(usize, usize, usize)>,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,
    direct_cost: Vec<Vec<f64>>,
    via_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,
    sorted_direct: Vec<Vec<usize>>,
    sorted_via: Vec<Vec<usize>>,
    avg_depot_round: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_prop: RefCell<Vec<usize>>,
    scratch_visited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_map: HashMap<String, (usize, usize)> = HashMap::new();
        let mut visited_map: HashMap<String, (usize, usize)> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let dom = task.variables()[var_id].domain_size();
            for value in 0..dom {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(obj) = extract_paren_arg(&fact_name, &["Atom loc(", "loc("]) {
                    loc_map.insert(obj, (var_id, value));
                } else if let Some(obj) = extract_paren_arg(&fact_name, &["Atom visited(", "visited("]) {
                    visited_map.insert(obj, (var_id, value));
                }
            }
        }

        let mut customers: Vec<String> = visited_map.keys().cloned().collect();
        customers.sort();
        let n = customers.len();
        if n == 0 {
            return Err("no customers found in visited facts".to_string().into());
        }

        let mut depot_candidates: Vec<&String> = loc_map
            .keys()
            .filter(|k| !visited_map.contains_key(*k))
            .collect();
        depot_candidates.sort();
        if depot_candidates.is_empty() {
            return Err("could not identify depot from loc/visited facts".to_string().into());
        }
        let depot_name = depot_candidates[0].clone();

        let mut place_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, c) in customers.iter().enumerate() {
            place_to_idx.insert(c.clone(), i);
        }
        let depot_idx = n;
        place_to_idx.insert(depot_name.clone(), depot_idx);

        let mut loc_facts: Vec<(usize, usize, usize)> = Vec::new();
        for (obj, &(var_id, value)) in &loc_map {
            let idx = if *obj == depot_name {
                depot_idx
            } else {
                *place_to_idx.get(obj).unwrap_or(&usize::MAX)
            };
            if idx != usize::MAX {
                loc_facts.push((var_id, value, idx));
            }
        }

        let mut visited_facts: Vec<(usize, usize, usize)> = Vec::new();
        for (i, c) in customers.iter().enumerate() {
            if let Some(&(var_id, value)) = visited_map.get(c) {
                visited_facts.push((var_id, value, i));
            }
        }

        let num_names: Vec<String> = task
            .numeric_variables()
            .iter()
            .map(|v| v.name().trim().to_string())
            .collect();

        let load_id = find_numeric_id(&num_names, &["PNE load()", "load()"])
            .ok_or_else(|| HeuristicBuildError::from("load() numeric variable not found".to_string()))?;
        let vehicles_id = find_numeric_id(&num_names, &["PNE vehicles()", "vehicles()"])
            .ok_or_else(|| HeuristicBuildError::from("vehicles() numeric variable not found".to_string()))?;

        let init_num = task.get_initial_numeric_state_values();
        let capacity = find_numeric_id(&num_names, &["PNE capacity()", "capacity()"])
            .map(|id| init_num[id])
            .unwrap_or(INF);
        let max_vehicles = find_numeric_id(&num_names, &["PNE max_vehicles()", "max_vehicles()"])
            .map(|id| init_num[id])
            .unwrap_or(INF);

        let mut demands = vec![0.0; n];
        for (i, c) in customers.iter().enumerate() {
            let exact_pne = format!("PNE demand({})", c);
            let exact_plain = format!("demand({})", c);
            if let Some(id) = find_numeric_id(&num_names, &[exact_pne.as_str(), exact_plain.as_str()]) {
                demands[i] = init_num[id];
            }
        }

        let mut direct_cost = vec![vec![INF; n]; n + 1];
        let mut via_cost = vec![vec![INF; n]; n];
        let mut return_cost = vec![INF; n];

        for op in task.get_operators() {
            let name = op.name();
            let parts: Vec<&str> = name.split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "visit" {
                if let Some(&from_idx) = place_to_idx.get(parts[1]) {
                    if let Some(&to_idx) = place_to_idx.get(parts[2]) {
                        if to_idx < n {
                            direct_cost[from_idx][to_idx] =
                                metric_operator_cost_from_initial_values(task, op);
                        }
                    }
                }
            } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
                if let Some(&from_idx) = place_to_idx.get(parts[1]) {
                    if let Some(&to_idx) = place_to_idx.get(parts[2]) {
                        if from_idx < n && to_idx < n {
                            via_cost[from_idx][to_idx] =
                                metric_operator_cost_from_initial_values(task, op);
                        }
                    }
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&from_idx) = place_to_idx.get(parts[1]) {
                    if from_idx < n {
                        return_cost[from_idx] =
                            metric_operator_cost_from_initial_values(task, op);
                    }
                }
            }
        }

        let any_missing_direct = (0..=n).any(|r| (0..n).any(|c| direct_cost[r][c] >= INF / 2.0));
        if any_missing_direct {
            for (id, var) in task.numeric_variables().iter().enumerate() {
                let raw = var.name().trim();
                if let Some(inner) = extract_travel_cost_arg(raw) {
                    let mut it = inner.split_whitespace();
                    if let (Some(from), Some(to)) = (it.next(), it.next()) {
                        if it.next().is_none() {
                            if let Some(&fi) = place_to_idx.get(from) {
                                if let Some(&ti) = place_to_idx.get(to) {
                                    if ti < n && direct_cost[fi][ti] >= INF / 2.0 {
                                        direct_cost[fi][ti] = init_num[id];
                                    } else if ti == depot_idx && fi < n && return_cost[fi] >= INF / 2.0 {
                                        return_cost[fi] = init_num[id];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for a in 0..n {
            if return_cost[a] >= INF / 2.0 {
                continue;
            }
            for b in 0..n {
                if via_cost[a][b] >= INF / 2.0 && direct_cost[depot_idx][b] < INF / 2.0 {
                    via_cost[a][b] = return_cost[a] + direct_cost[depot_idx][b];
                }
            }
        }

        if n > 0 && (0..n).all(|c| direct_cost[depot_idx][c] >= INF / 2.0) {
            return Err("Could not extract any depot outbound travel costs".to_string().into());
        }

        let mut sorted_direct = Vec::with_capacity(n + 1);
        for r in 0..=n {
            let mut entries: Vec<(f64, usize)> = (0..n)
                .filter_map(|c| {
                    let d = direct_cost[r][c];
                    if d < INF / 2.0 {
                        Some((d, c))
                    } else {
                        None
                    }
                })
                .collect();
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            sorted_direct.push(entries.into_iter().map(|(_, c)| c).collect());
        }

        let mut sorted_via = Vec::with_capacity(n);
        for r in 0..n {
            let mut entries: Vec<(f64, usize)> = (0..n)
                .filter_map(|c| {
                    let d = via_cost[r][c];
                    if d < INF / 2.0 {
                        Some((d, c))
                    } else {
                        None
                    }
                })
                .collect();
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            sorted_via.push(entries.into_iter().map(|(_, c)| c).collect());
        }

        let mut sum_depot = 0.0;
        let mut cnt_depot = 0usize;
        for c in 0..n {
            if direct_cost[depot_idx][c] < INF / 2.0 && return_cost[c] < INF / 2.0 {
                sum_depot += direct_cost[depot_idx][c] + return_cost[c];
                cnt_depot += 1;
            }
        }
        let avg_depot_round = if cnt_depot > 0 {
            sum_depot / cnt_depot as f64
        } else {
            0.0
        };

        Ok(Self {
            n_customers: n,
            depot_idx,
            loc_facts,
            visited_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            direct_cost,
            via_cost,
            return_cost,
            sorted_direct,
            sorted_via,
            avg_depot_round,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_prop: RefCell::new(vec![0usize; task.variables().len()]),
            scratch_visited: RefCell::new(vec![false; n]),
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut numeric = self.scratch_numeric.borrow_mut();
        registry
            .view(state)
            .fill_numeric(&mut *numeric)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;
        let load = numeric[self.load_id];
        let vehicles = numeric[self.vehicles_id];
        drop(numeric);

        let mut prop = self.scratch_prop.borrow_mut();
        state.fill_state(registry, &mut *prop);

        let mut current = usize::MAX;
        for (var_id, value, node_idx) in &self.loc_facts {
            if prop[*var_id] == *value {
                current = *node_idx;
                break;
            }
        }
        if current == usize::MAX {
            drop(prop);
            return Ok(LARGE_PENALTY);
        }

        let mut visited = self.scratch_visited.borrow_mut();
        for v in visited.iter_mut() {
            *v = false;
        }
        let mut unvisited_count = 0;
        for (var_id, value, cust_idx) in &self.visited_facts {
            if prop[*var_id] == *value {
                visited[*cust_idx] = true;
            } else {
                unvisited_count += 1;
            }
        }
        drop(prop);

        if unvisited_count == 0 {
            if current != self.depot_idx {
                let cost = self.return_cost[current];
                if cost >= INF / 2.0 {
                    return Ok(LARGE_PENALTY);
                }
                return Ok(cost);
            } else {
                return Ok(0.0);
            }
        }

        let mut remaining_demand = 0.0;
        for i in 0..self.n_customers {
            if !visited[i] {
                remaining_demand += self.demands[i];
            }
        }

        let residual_capacity = (self.capacity - load).max(0.0);
        let extra_vehicles = if self.max_vehicles >= INF / 2.0 {
            INF
        } else {
            (self.max_vehicles - vehicles).max(0.0)
        };
        let available_capacity = if extra_vehicles >= INF / 2.0 || self.capacity >= INF / 2.0 {
            INF
        } else {
            residual_capacity + extra_vehicles * self.capacity
        };

        if remaining_demand > available_capacity + 1e-9 {
            return Ok(LARGE_PENALTY);
        }

        let mut h_capacity = 0.0;
        if self.capacity < INF / 2.0 {
            let remaining_for_extra = (remaining_demand - residual_capacity).max(0.0);
            if remaining_for_extra > 1e-9 {
                let extra_needed = (remaining_for_extra / self.capacity).ceil();
                h_capacity = extra_needed * self.avg_depot_round * 0.25;
            }
        }

        let mut h_route = 0.0;
        let mut cur_node = current;
        let mut cur_load = load;
        let mut cur_vehicles = vehicles;
        let mut remaining = unvisited_count;

        while remaining > 0 {
            let mut selected: Option<(usize, f64, bool)> = None;

            if cur_node == self.depot_idx {
                if let Some(&to) = self.sorted_direct[cur_node]
                    .iter()
                    .find(|&&c| !visited[c] && cur_load + self.demands[c] <= self.capacity + 1e-9)
                {
                    selected = Some((to, self.direct_cost[cur_node][to], false));
                }
            } else {
                let direct = self.sorted_direct[cur_node]
                    .iter()
                    .find(|&&c| !visited[c] && cur_load + self.demands[c] <= self.capacity + 1e-9)
                    .map(|&c| (c, self.direct_cost[cur_node][c], false));

                let via = self.sorted_via[cur_node]
                    .iter()
                    .find(|&&c| !visited[c] && cur_vehicles + 1.0 <= self.max_vehicles + 1e-9)
                    .map(|&c| (c, self.via_cost[cur_node][c], true));

                selected = match (direct, via) {
                    (Some(d), Some(v)) => if d.1 <= v.1 { Some(d) } else { Some(v) },
                    (Some(d), None) => Some(d),
                    (None, Some(v)) => Some(v),
                    (None, None) => None,
                };
            }

            if let Some((to, cost, is_via)) = selected {
                h_route += cost;
                visited[to] = true;
                remaining -= 1;
                if is_via {
                    cur_load = self.demands[to];
                    cur_vehicles += 1.0;
                } else {
                    cur_load += self.demands[to];
                }
                cur_node = to;
            } else {
                return Ok(LARGE_PENALTY);
            }
        }

        if cur_node != self.depot_idx {
            let cost = self.return_cost[cur_node];
            if cost >= INF / 2.0 {
                return Ok(LARGE_PENALTY);
            }
            h_route += cost;
        }

        Ok(h_route + h_capacity)
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
