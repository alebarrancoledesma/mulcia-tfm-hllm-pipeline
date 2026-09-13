use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::state::ExplicitFact;

use planforge_search::config::HeuristicSpec;

use planforge_search::evaluation::{
    EvaluationError,
    EvaluationState,
    Heuristic,
    metric_operator_cost_from_initial_values,
};

use planforge_search::heuristic_factory::{
    ExternalHeuristic,
    HeuristicBuildError,
    RequiredBackend,
    TaskRequirements,
};

const INF: f64 = 1e30;
const LARGE_PENALTY: f64 = 1e12;
const EPS: f64 = 1e-6;

fn parse_single_arg(name: &str, prefix: &str) -> Option<String> {
    let start = name.find(prefix)?;
    let rest = &name[start + prefix.len()..];
    let end = rest.find(')')?;
    Some(rest[..end].trim().to_string())
}

fn parse_two_args(s: &str) -> Option<(String, String)> {
    let mut parts = s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|p| !p.is_empty());
    let a = parts.next()?.to_string();
    let b = parts.next()?.to_string();
    Some((a, b))
}

fn scan_facts(
    task: &dyn AbstractNumericTask,
) -> Result<(HashMap<String, ExplicitFact>, HashMap<String, ExplicitFact>), HeuristicBuildError> {
    let mut loc = HashMap::new();
    let mut visited = HashMap::new();

    for op in task.get_operators() {
        for fact in op.preconditions() {
            let name = task.get_fact_name(*fact);
            if name.starts_with("Atom loc(") {
                if let Some(obj) = parse_single_arg(&name, "loc(") {
                    loc.entry(obj).or_insert(*fact);
                }
            } else if name.starts_with("Atom visited(") {
                if let Some(obj) = parse_single_arg(&name, "visited(") {
                    visited.entry(obj).or_insert(*fact);
                }
            }
        }
        for fact in op.effects() {
            let name = task.get_fact_name(*fact);
            if name.starts_with("Atom loc(") {
                if let Some(obj) = parse_single_arg(&name, "loc(") {
                    loc.entry(obj).or_insert(*fact);
                }
            } else if name.starts_with("Atom visited(") {
                if let Some(obj) = parse_single_arg(&name, "visited(") {
                    visited.entry(obj).or_insert(*fact);
                }
            }
        }
    }

    for i in 0..task.get_num_goals() {
        let fact = task.get_goal_fact(i);
        let name = task.get_fact_name(fact);
        if name.starts_with("Atom visited(") {
            if let Some(obj) = parse_single_arg(&name, "visited(") {
                visited.entry(obj).or_insert(fact);
            }
        }
    }

    if visited.is_empty() {
        return Err("Could not discover any visited() facts".to_string().into());
    }
    Ok((loc, visited))
}

struct LlmHeuristic {
    depot_idx: usize,
    n_customers: usize,
    node_count: usize,
    visited_facts: Vec<ExplicitFact>,
    loc_facts: Vec<ExplicitFact>,
    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    direct: Vec<Vec<f64>>,
    via_depot: Vec<Vec<f64>>,
    to_depot: Vec<f64>,
    from_depot: Vec<f64>,
    nearest_direct: Vec<Vec<usize>>,
    nearest_via: Vec<Vec<usize>>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_visited: RefCell<Vec<bool>>,
    scratch_sim_visited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let (loc_map, visited_map) = scan_facts(task)?;

        let mut customer_names: Vec<String> = visited_map.keys().cloned().collect();
        customer_names.sort();

        let visited_set: HashSet<String> = customer_names.iter().cloned().collect();
        let depot_candidates: Vec<String> = loc_map
            .keys()
            .filter(|name| !visited_set.contains(*name))
            .cloned()
            .collect();
        let depot_name = depot_candidates.first().cloned().ok_or_else(|| {
            HeuristicBuildError::from("Could not identify depot from loc/visited facts".to_string())
        })?;

        let n = customer_names.len();
        let mut node_names = customer_names.clone();
        node_names.push(depot_name.clone());
        let depot_idx = n;
        let node_count = node_names.len();

        let node_to_idx: HashMap<String, usize> = node_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let visited_facts = customer_names
            .iter()
            .map(|name| {
                visited_map.get(name).cloned().ok_or_else(|| {
                    HeuristicBuildError::from(format!("Missing visited fact for customer {}", name))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let loc_facts = node_names
            .iter()
            .map(|name| {
                loc_map.get(name).cloned().ok_or_else(|| {
                    HeuristicBuildError::from(format!("Missing loc fact for {}", name))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let initial_values = task.get_initial_numeric_state_values();

        let mut direct = vec![vec![INF; node_count]; node_count];
        let mut via_depot = vec![vec![INF; n]; n];
        let mut to_depot = vec![INF; n];
        let mut from_depot = vec![INF; n];
        let mut demands = vec![0.0; n];

        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity: f64 = f64::INFINITY;
        let mut max_vehicles: f64 = f64::INFINITY;

        for var in task.numeric_variables() {
            let name = var.name();
            let id = var.id();
            let value = initial_values.get(id).copied().unwrap_or(0.0);

            if name.contains("load()") || name.ends_with("load()") {
                load_id = Some(id);
            }
            if name.contains("vehicles()") || name.ends_with("vehicles()") {
                vehicles_id = Some(id);
            }
            if name.contains("capacity") {
                capacity = value;
            }
            if name.contains("max_vehicles") || name.contains("max-vehicles") {
                max_vehicles = value;
            }

            if let Some(obj) = parse_single_arg(&name, "demand(") {
                if let Some(&idx) = node_to_idx.get(&obj) {
                    if idx < n {
                        demands[idx] = value;
                    }
                }
            }

            if let Some(inner) = parse_single_arg(&name, "travel-cost-via-depot(") {
                if let Some((a, b)) = parse_two_args(&inner) {
                    if let (Some(&i), Some(&j)) = (node_to_idx.get(&a), node_to_idx.get(&b)) {
                        if i < n && j < n && via_depot[i][j] >= INF / 2.0 {
                            via_depot[i][j] = value;
                        }
                    }
                }
            } else if let Some(inner) = parse_single_arg(&name, "travel-cost(") {
                if let Some((a, b)) = parse_two_args(&inner) {
                    if let (Some(&i), Some(&j)) = (node_to_idx.get(&a), node_to_idx.get(&b)) {
                        if direct[i][j] >= INF / 2.0 {
                            direct[i][j] = value;
                        }
                    }
                }
            }
        }

        if load_id.is_none() || vehicles_id.is_none() {
            return Err(HeuristicBuildError::from(
                "Could not discover load() and vehicles() numeric variables".to_string(),
            ));
        }

        for op in task.get_operators() {
            let name = op.name();
            let tokens: Vec<&str> = name.split_whitespace().collect();

            if tokens.len() == 3 && tokens[0] == "visit" {
                let from = tokens[1];
                let to = tokens[2];
                let cost = metric_operator_cost_from_initial_values(task, op);
                if let (Some(&fi), Some(&ti)) = (node_to_idx.get(from), node_to_idx.get(to)) {
                    direct[fi][ti] = cost;
                    if fi == depot_idx {
                        from_depot[ti] = cost;
                    }
                }
            } else if tokens.len() == 3 && tokens[0] == "visit-via-depot" {
                let from = tokens[1];
                let to = tokens[2];
                let cost = metric_operator_cost_from_initial_values(task, op);
                if let (Some(&fi), Some(&ti)) = (node_to_idx.get(from), node_to_idx.get(to)) {
                    if fi < n && ti < n {
                        via_depot[fi][ti] = cost;
                    }
                }
            } else if tokens.len() == 2 && tokens[0] == "return-to-depot" {
                let from = tokens[1];
                let cost = metric_operator_cost_from_initial_values(task, op);
                if let Some(&fi) = node_to_idx.get(from) {
                    if fi < n {
                        to_depot[fi] = cost;
                    }
                }
            }
        }

        // Fill missing essential arcs conservatively without assuming symmetry.
        for i in 0..n {
            if to_depot[i] >= INF / 2.0 && direct[i][depot_idx] < INF / 2.0 {
                to_depot[i] = direct[i][depot_idx];
            }
            if from_depot[i] >= INF / 2.0 && direct[depot_idx][i] < INF / 2.0 {
                from_depot[i] = direct[depot_idx][i];
            }
            direct[i][depot_idx] = to_depot[i].min(direct[i][depot_idx]);
            direct[depot_idx][i] = from_depot[i].min(direct[depot_idx][i]);
        }

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if direct[i][j] >= INF / 2.0 && via_depot[i][j] < INF / 2.0 {
                    direct[i][j] = via_depot[i][j];
                } else if direct[i][j] >= INF / 2.0
                    && to_depot[i] < INF / 2.0
                    && from_depot[j] < INF / 2.0
                {
                    direct[i][j] = to_depot[i] + from_depot[j];
                }
                if via_depot[i][j] >= INF / 2.0
                    && to_depot[i] < INF / 2.0
                    && from_depot[j] < INF / 2.0
                {
                    via_depot[i][j] = to_depot[i] + from_depot[j];
                }
            }
        }

        let mut nearest_direct = vec![Vec::new(); node_count];
        for u in 0..node_count {
            let mut list: Vec<usize> = (0..n).filter(|&v| v != u).collect();
            list.sort_by(|&a, &b| {
                let ca = direct[u][a];
                let cb = direct[u][b];
                if ca < cb {
                    Ordering::Less
                } else if ca > cb {
                    Ordering::Greater
                } else {
                    a.cmp(&b)
                }
            });
            nearest_direct[u] = list;
        }

        let mut nearest_via = vec![Vec::new(); n];
        for i in 0..n {
            let mut list: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            list.sort_by(|&a, &b| {
                let ca = via_depot[i][a];
                let cb = via_depot[i][b];
                if ca < cb {
                    Ordering::Less
                } else if ca > cb {
                    Ordering::Greater
                } else {
                    a.cmp(&b)
                }
            });
            nearest_via[i] = list;
        }

        Ok(Self {
            depot_idx,
            n_customers: n,
            node_count,
            visited_facts,
            loc_facts,
            demands,
            capacity,
            max_vehicles,
            direct,
            via_depot,
            to_depot,
            from_depot,
            nearest_direct,
            nearest_via,
            load_id,
            vehicles_id,
            scratch_numeric: RefCell::new(vec![0.0; 4]),
            scratch_visited: RefCell::new(vec![false; n]),
            scratch_sim_visited: RefCell::new(vec![false; n]),
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

        let (load, vehicles) = {
            let mut nums = self.scratch_numeric.borrow_mut();
            nums.clear();
            registry.view(state).fill_numeric(&mut nums);
            let load = match self.load_id {
                Some(id) => nums.get(id).copied().unwrap_or(0.0),
                None => 0.0,
            };
            let vehicles = match self.vehicles_id {
                Some(id) => nums.get(id).copied().unwrap_or(1.0),
                None => 1.0,
            };
            (load, vehicles)
        };

        {
            let mut actual = self.scratch_visited.borrow_mut();
            for i in 0..self.n_customers {
                actual[i] = self.visited_facts[i].is_hold(registry.view(state));
            }
        }

        let mut cur_node = self.depot_idx;
        for (idx, fact) in self.loc_facts.iter().enumerate() {
            if fact.is_hold(registry.view(state)) {
                cur_node = idx;
                break;
            }
        }

        let mut remaining_demand = 0.0;
        let mut unvisited_count = 0usize;
        {
            let actual = self.scratch_visited.borrow();
            for i in 0..self.n_customers {
                if !actual[i] {
                    remaining_demand += self.demands[i];
                    unvisited_count += 1;
                }
            }
        }

        if unvisited_count == 0 {
            if cur_node == self.depot_idx {
                return Ok(0.0);
            }
            let ret = self.to_depot[cur_node];
            return if ret < INF / 2.0 { Ok(ret) } else { Ok(0.0) };
        }

        let capacity = self.capacity.max(EPS);
        let residual_capacity = (capacity - load).max(0.0);
        let remaining_vehicles = (self.max_vehicles - vehicles).max(0.0);
        let total_capacity_left = residual_capacity + remaining_vehicles * capacity;

        if remaining_demand > total_capacity_left + EPS {
            return Ok(LARGE_PENALTY + remaining_demand);
        }

        let h_capacity_component = (remaining_demand / capacity) * 2.0;
        let est_required_vehicles = (remaining_demand / capacity).ceil();
        let h_vehicle_component = est_required_vehicles * 3.0;
        let mut h = h_capacity_component + h_vehicle_component;

        {
            let actual = self.scratch_visited.borrow();
            let mut sim_visited = self.scratch_sim_visited.borrow_mut();
            for i in 0..self.n_customers {
                sim_visited[i] = actual[i];
            }

            let mut sim_cur = cur_node;
            let mut sim_load = load;
            let mut sim_vehicles = vehicles;
            let mut remaining = unvisited_count;
            let mut route_est = 0.0;

            while remaining > 0 {
                let mut chosen = None;

                if sim_cur == self.depot_idx {
                    for &cand in &self.nearest_direct[sim_cur] {
                        if sim_visited[cand] {
                            continue;
                        }
                        let dem = self.demands[cand];
                        if sim_load + dem <= capacity + EPS {
                            let cost = self.direct[sim_cur][cand];
                            if cost < INF / 2.0 {
                                chosen = Some(cand);
                                route_est += cost;
                                sim_load += dem;
                                sim_cur = cand;
                                break;
                            }
                        }
                    }
                } else {
                    for &cand in &self.nearest_direct[sim_cur] {
                        if sim_visited[cand] {
                            continue;
                        }
                        let dem = self.demands[cand];
                        if sim_load + dem <= capacity + EPS {
                            let cost = self.direct[sim_cur][cand];
                            if cost < INF / 2.0 {
                                chosen = Some(cand);
                                route_est += cost;
                                sim_load += dem;
                                sim_cur = cand;
                                break;
                            }
                        }
                    }

                    if chosen.is_none() && sim_vehicles + 1.0 <= self.max_vehicles + EPS {
                        let from_idx = sim_cur;
                        for &cand in &self.nearest_via[from_idx] {
                            if sim_visited[cand] {
                                continue;
                            }
                            let dem = self.demands[cand];
                            if dem <= capacity + EPS {
                                let cost = self.via_depot[from_idx][cand];
                                if cost < INF / 2.0 {
                                    chosen = Some(cand);
                                    route_est += cost;
                                    sim_vehicles += 1.0;
                                    sim_load = dem;
                                    sim_cur = cand;
                                    break;
                                }
                            }
                        }
                    }
                }

                if let Some(c) = chosen {
                    sim_visited[c] = true;
                    remaining -= 1;
                } else {
                    return Ok(LARGE_PENALTY + remaining as f64);
                }
            }

            if sim_cur != self.depot_idx {
                let ret = self.to_depot[sim_cur];
                if ret < INF / 2.0 {
                    route_est += ret;
                }
            }

            h += route_est;
        }

        Ok(h.min(LARGE_PENALTY))
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
