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
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const INF: f64 = 1e30_f64;
const LARGE_PENALTY: f64 = 1e15_f64;

struct Scratch {
    unvisited: Vec<usize>,
    in_rem: Vec<bool>,
    numeric: Vec<f64>,
}

fn parse_fact_arg(fact_name: &str, predicate: &str) -> Option<String> {
    let needle = format!("{predicate}(");
    let idx = fact_name.find(&needle)?;
    let rest = &fact_name[idx + needle.len()..];
    let end = rest.find(')')?;
    let arg = rest[..end].trim();
    if arg.is_empty() {
        None
    } else {
        Some(arg.to_string())
    }
}

fn find_numeric_by_exact_names(task: &dyn AbstractNumericTask, names: &[&str]) -> Option<usize> {
    for (id, var) in task.numeric_variables().iter().enumerate() {
        let n = var.name().trim();
        if names.iter().any(|cand| n == *cand) {
            return Some(id);
        }
    }
    None
}

fn parse_argv_args(name: &str, prefix: &str) -> Option<(String, String)> {
    let idx = name.find(prefix)?;
    let after = &name[idx + prefix.len()..];
    if !after.starts_with('(') {
        return None;
    }
    let end = after.find(')')?;
    let args = &after[1..end];
    let mut tokens = args
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty());
    let a = tokens.next()?;
    let b = tokens.next()?;
    if tokens.next().is_some() {
        return None;
    }
    Some((a.to_string(), b.to_string()))
}

struct LlmHeuristic {
    n: usize,
    depot_idx: usize,
    loc_facts: Vec<(String, ExplicitFact)>,
    visited_facts: Vec<Option<ExplicitFact>>,
    place_index: HashMap<String, usize>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    capacity_value: f64,
    max_vehicles_value: f64,
    demands: Vec<f64>,
    direct: Vec<Vec<f64>>,
    via_depot: Vec<Vec<f64>>,
    to_depot: Vec<f64>,
    nearest_direct: Vec<Vec<usize>>,
    nearest_via: Vec<Vec<usize>>,
    avg_depot_cost: f64,
    scratch: RefCell<Scratch>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_facts: Vec<(String, ExplicitFact)> = Vec::new();
        let mut visited_map: HashMap<String, ExplicitFact> = HashMap::new();

        let variables = task.variables();
        for var_id in 0..variables.len() {
            let domain_size = variables[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = format!("{:?}", task.get_fact_name(&fact));
                if let Some(arg) = parse_fact_arg(&fact_name, "loc") {
                    loc_facts.push((arg, fact));
                } else if let Some(arg) = parse_fact_arg(&fact_name, "visited") {
                    if !visited_map.contains_key(&arg) {
                        visited_map.insert(arg, fact);
                    }
                }
            }
        }

        let visited_names: HashSet<String> = visited_map.keys().cloned().collect();
        let mut depot_name: Option<String> = None;
        let mut customer_name_set: HashSet<String> = HashSet::new();

        for (p, _) in &loc_facts {
            if visited_names.contains(p) {
                customer_name_set.insert(p.clone());
            } else if depot_name.is_none() {
                depot_name = Some(p.clone());
            }
        }

        for name in visited_names.iter() {
            customer_name_set.insert(name.clone());
        }

        let depot_name = depot_name
            .or_else(|| loc_facts.first().map(|(p, _)| p.clone()))
            .ok_or_else(|| HeuristicBuildError::from("missing depot loc fact".to_string()))?;

        let mut customer_names: Vec<String> = customer_name_set.into_iter().collect();
        customer_names.sort();
        let n = customer_names.len();
        let depot_idx = n;

        let mut place_index: HashMap<String, usize> = HashMap::new();
        for (i, name) in customer_names.iter().enumerate() {
            place_index.insert(name.clone(), i);
        }
        place_index.insert(depot_name.clone(), depot_idx);

        let visited_facts: Vec<Option<ExplicitFact>> = customer_names
            .iter()
            .map(|name| visited_map.remove(name))
            .collect();

        let initial_numeric = task.get_initial_numeric_state_values();
        let load_id = find_numeric_by_exact_names(task, &["load()", "PNE load()"]);
        let vehicles_id = find_numeric_by_exact_names(task, &["vehicles()", "PNE vehicles()"]);

        let capacity_value = find_numeric_by_exact_names(task, &["capacity()", "PNE capacity()"])
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(f64::MAX);

        let max_vehicles_value =
            find_numeric_by_exact_names(task, &["max_vehicles()", "PNE max_vehicles()"])
                .and_then(|id| initial_numeric.get(id).copied())
                .filter(|v| v.is_finite())
                .unwrap_or(f64::MAX);

        let mut demands = vec![0.0_f64; n];
        for (i, cname) in customer_names.iter().enumerate() {
            let c1 = format!("demand({})", cname);
            let c2 = format!("PNE demand({})", cname);
            let cands = [c1.as_str(), c2.as_str()];
            if let Some(v) = find_numeric_by_exact_names(task, &cands)
                .and_then(|id| initial_numeric.get(id).copied())
                .filter(|v| v.is_finite())
            {
                demands[i] = v;
            }
        }

        let mut direct = vec![vec![INF; n + 1]; n + 1];
        let mut via_depot = vec![vec![INF; n]; n];
        let mut to_depot = vec![INF; n];

        // Phase A: primary extraction from grounded operators.
        for op in task.get_operators() {
            let name = op.name().trim();
            let tokens: Vec<&str> = name.split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() {
                continue;
            }

            if tokens.len() >= 3 && tokens[0] == "visit" {
                if let (Some(&a), Some(&b)) =
                    (place_index.get(tokens[1]), place_index.get(tokens[2]))
                {
                    if cost < direct[a][b] {
                        direct[a][b] = cost;
                    }
                }
            } else if tokens.len() >= 3 && tokens[0] == "visit-via-depot" {
                if let (Some(&a), Some(&b)) =
                    (place_index.get(tokens[1]), place_index.get(tokens[2]))
                {
                    if a < n && b < n && cost < via_depot[a][b] {
                        via_depot[a][b] = cost;
                    }
                }
            } else if tokens.len() >= 2 && tokens[0] == "return-to-depot" {
                if let Some(&a) = place_index.get(tokens[1]) {
                    if a < n {
                        to_depot[a] = cost;
                        direct[a][depot_idx] = cost;
                    }
                }
            }
        }

        // Phase B: fallback extraction from explicit travel-cost numeric names.
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let raw = var.name();
            if let Some((a, b)) = parse_argv_args(raw, "travel-cost-via-depot(") {
                if let (Some(&ai), Some(&bi)) = (place_index.get(&a), place_index.get(&b)) {
                    if ai < n && bi < n {
                        if let Some(v) = initial_numeric.get(id).copied().filter(|v| v.is_finite()) {
                            if v < via_depot[ai][bi] {
                                via_depot[ai][bi] = v;
                            }
                        }
                    }
                }
            } else if let Some((a, b)) = parse_argv_args(raw, "travel-cost(") {
                if let (Some(&ai), Some(&bi)) = (place_index.get(&a), place_index.get(&b)) {
                    if let Some(v) = initial_numeric.get(id).copied().filter(|v| v.is_finite()) {
                        if v < direct[ai][bi] {
                            direct[ai][bi] = v;
                        }
                    }
                }
            }
        }

        // Phase C: conservative completion for still-missing arcs via the depot.
        for i in 0..n {
            if direct[i][depot_idx] >= INF && to_depot[i] < INF {
                direct[i][depot_idx] = to_depot[i];
            }
            for j in 0..n {
                if i == j {
                    continue;
                }
                if direct[i][j] >= INF
                    && direct[i][depot_idx] < INF
                    && direct[depot_idx][j] < INF
                {
                    direct[i][j] = direct[i][depot_idx] + direct[depot_idx][j];
                }
                if via_depot[i][j] >= INF
                    && direct[i][depot_idx] < INF
                    && direct[depot_idx][j] < INF
                {
                    via_depot[i][j] = direct[i][depot_idx] + direct[depot_idx][j];
                }
            }
        }

        if n > 0 {
            let depot_out_known = (0..n).filter(|&c| direct[depot_idx][c] < INF).count();
            if depot_out_known == 0 {
                return Err(HeuristicBuildError::from(
                    "could not recover any depot outbound travel cost from grounded operators".to_string(),
                ));
            }
        }

        let mut nearest_direct: Vec<Vec<usize>> = Vec::with_capacity(n + 1);
        for p in 0..=n {
            let mut list: Vec<usize> = (0..n).filter(|&c| direct[p][c] < INF).collect();
            list.sort_by(|a, b| {
                direct[p][*a]
                    .partial_cmp(&direct[p][*b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_direct.push(list);
        }

        let mut nearest_via: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut list: Vec<usize> = (0..n).filter(|&j| via_depot[i][j] < INF).collect();
            list.sort_by(|a, b| {
                via_depot[i][*a]
                    .partial_cmp(&via_depot[i][*b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_via.push(list);
        }

        let avg_depot_cost = {
            let mut sum = 0.0_f64;
            let mut count = 0usize;
            for c in 0..n {
                if direct[depot_idx][c] < INF {
                    sum += direct[depot_idx][c];
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                0.0_f64
            }
        };

        let scratch = Scratch {
            unvisited: Vec::with_capacity(n),
            in_rem: vec![false; n],
            numeric: vec![0.0_f64; task.numeric_variables().len()],
        };

        Ok(Self {
            n,
            depot_idx,
            loc_facts,
            visited_facts,
            place_index,
            load_id,
            vehicles_id,
            capacity_value,
            max_vehicles_value,
            demands,
            direct,
            via_depot,
            to_depot,
            nearest_direct,
            nearest_via,
            avg_depot_cost,
            scratch: RefCell::new(scratch),
        })
    }
}

impl Heuristic for LlmHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if eval_state.is_goal() {
            return Ok(0.0_f64);
        }

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let direct = &self.direct;
        let via_depot = &self.via_depot;
        let nearest_direct = &self.nearest_direct;
        let nearest_via = &self.nearest_via;
        let demands = &self.demands;
        let to_depot = &self.to_depot;
        let loc_facts = &self.loc_facts;
        let visited_facts = &self.visited_facts;
        let place_index = &self.place_index;
        let n = self.n;
        let depot_idx = self.depot_idx;
        let capacity = self.capacity_value;
        let max_vehicles = self.max_vehicles_value;
        let avg_depot_cost = self.avg_depot_cost;
        let load_id = self.load_id;
        let vehicles_id = self.vehicles_id;

        let mut sc = self.scratch.borrow_mut();

        registry
            .view(state)
            .fill_numeric(&mut sc.numeric)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

        let load = load_id.map(|id| sc.numeric[id]).unwrap_or(0.0_f64);
        let vehicles_used = vehicles_id
            .map(|id| sc.numeric[id])
            .unwrap_or(1.0_f64);

        let mut current_idx: Option<usize> = None;
        for (place, fact) in loc_facts {
            if fact.is_hold(registry.view(state)) {
                current_idx = place_index.get(place).copied();
                break;
            }
        }
        let current_idx = current_idx.unwrap_or(depot_idx);

        sc.unvisited.clear();
        for i in 0..n {
            let visited = visited_facts[i]
                .as_ref()
                .map(|f| f.is_hold(registry.view(state)))
                .unwrap_or(false);
            if !visited {
                sc.unvisited.push(i);
            }
        }

        if sc.unvisited.is_empty() {
            if current_idx == depot_idx {
                return Ok(0.0_f64);
            }
            let ret = to_depot[current_idx];
            return Ok(if ret < INF { ret } else { LARGE_PENALTY });
        }

        let remaining_demand: f64 = sc.unvisited.iter().map(|&c| demands[c]).sum();
        let current_rem_cap = if capacity.is_finite() {
            (capacity - load).max(0.0_f64)
        } else {
            f64::MAX
        };

        let available_capacity = if capacity.is_finite() {
            if max_vehicles.is_finite() {
                let add = (max_vehicles - vehicles_used).max(0.0_f64) * capacity;
                let total = current_rem_cap + if add.is_finite() { add } else { 0.0 };
                if total.is_finite() {
                    total
                } else {
                    f64::MAX
                }
            } else {
                f64::MAX
            }
        } else {
            f64::MAX
        };

        if remaining_demand > available_capacity + 1e-9_f64 {
            return Ok(LARGE_PENALTY);
        }

        let rem_cap = if capacity.is_finite() {
            capacity - load
        } else {
            f64::MAX
        };

        let can_direct = sc
            .unvisited
            .iter()
            .any(|&c| direct[current_idx][c] < INF && demands[c] <= rem_cap);

        let can_via = current_idx != depot_idx
            && max_vehicles.is_finite()
            && vehicles_used <= max_vehicles - 1.0_f64
            && sc
                .unvisited
                .iter()
                .any(|&c| via_depot[current_idx][c] < INF);

        if !can_direct && !can_via {
            return Ok(LARGE_PENALTY);
        }

        let h_resource = if capacity.is_finite() && capacity > 0.0_f64 {
            let after_current = (remaining_demand - current_rem_cap).max(0.0_f64);
            let extra_routes = (after_current / capacity).ceil();
            extra_routes * avg_depot_cost
        } else {
            0.0_f64
        };

        for flag in sc.in_rem.iter_mut() {
            *flag = false;
        }
        for &c in &sc.unvisited {
            sc.in_rem[c] = true;
        }
        let mut remaining_count = sc.unvisited.len();

        let mut cur = current_idx;
        let mut cur_load = load;
        let mut used_veh = vehicles_used;
        let mut route_cost = 0.0_f64;

        while remaining_count > 0 {
            let cap_rem = if capacity.is_finite() {
                capacity - cur_load
            } else {
                f64::MAX
            };

            let mut best_direct: Option<(f64, usize)> = None;
            for &c in &nearest_direct[cur] {
                if sc.in_rem[c] {
                    if demands[c] <= cap_rem {
                        best_direct = Some((direct[cur][c], c));
                        break;
                    }
                }
            }

            let mut best_via: Option<(f64, usize)> = None;
            let via_allowed = cur != depot_idx
                && max_vehicles.is_finite()
                && used_veh <= max_vehicles - 1.0_f64;
            if via_allowed {
                for &c in &nearest_via[cur] {
                    if sc.in_rem[c] {
                        best_via = Some((via_depot[cur][c], c));
                        break;
                    }
                }
            }

            let (cost, chosen, via_used): (f64, usize, bool) =
                match (best_direct, best_via) {
                    (Some((dc, di)), Some((vc, vi))) => {
                        if dc <= vc {
                            (dc, di, false)
                        } else {
                            (vc, vi, true)
                        }
                    }
                    (Some((dc, di)), None) => (dc, di, false),
                    (None, Some((vc, vi))) => (vc, vi, true),
                    (None, None) => return Ok(LARGE_PENALTY),
                };

            sc.in_rem[chosen] = false;
            remaining_count -= 1;
            route_cost += cost;

            if via_used {
                used_veh += 1.0_f64;
                cur_load = demands[chosen];
            } else {
                cur_load += demands[chosen];
            }
            cur = chosen;
        }

        if cur != depot_idx {
            let ret = to_depot[cur];
            if ret < INF {
                route_cost += ret;
            } else {
                return Ok(LARGE_PENALTY);
            }
        }

        let h = route_cost + h_resource;
        Ok(h.max(0.0_f64))
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
                .into(),
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
