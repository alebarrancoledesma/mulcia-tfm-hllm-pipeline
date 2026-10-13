use std::collections::HashMap;
use std::cmp::Ordering;

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
struct CustomerInfo {
    place_idx: usize,
    var_id: usize,
    true_value: usize,
}

#[derive(Clone, Copy)]
struct CwRoute {
    first: usize,
    last: usize,
    load: f64,
    count: usize,
}

struct LlmHeuristic {
    depot: usize,
    loc_var_id: usize,
    loc_value_for_place: Vec<usize>,
    customer_infos: Vec<CustomerInfo>,
    direct: Vec<Vec<f64>>,
    start_cost: Vec<f64>,
    return_cost: Vec<f64>,
    demand_by_place: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    n_prop: usize,
    n_num: usize,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut loc_var_id: Option<usize> = None;
        let mut loc_values: HashMap<String, usize> = HashMap::new();
        let mut visited_infos: HashMap<String, (usize, usize)> = HashMap::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact).to_string();
                let name = name.trim();

                if let Some(rest) = name.strip_prefix("Atom loc(") {
                    let place = rest.trim_end_matches(')').trim().to_string();
                    loc_var_id = Some(var_id);
                    loc_values.insert(place, value);
                } else if let Some(rest) = name.strip_prefix("Atom visited(") {
                    let customer = rest.trim_end_matches(')').trim().to_string();
                    visited_infos.entry(customer).or_insert((var_id, value));
                }
            }
        }

        let loc_var_id = loc_var_id.ok_or_else(|| -> HeuristicBuildError {
            "missing loc fact".to_string().into()
        })?;

        let mut places: Vec<String> = loc_values.keys().cloned().collect();
        places.sort();

        let mut place_idx: HashMap<String, usize> = HashMap::new();
        for (idx, place) in places.iter().enumerate() {
            place_idx.insert(place.clone(), idx);
        }

        let n_places = places.len();

        let mut customer_names: Vec<String> = visited_infos.keys().cloned().collect();
        customer_names.sort();

        let depot_candidates: Vec<&String> = places
            .iter()
            .filter(|p| !visited_infos.contains_key(p.as_str()))
            .collect();

        let depot_name: String = if depot_candidates.len() == 1 {
            depot_candidates[0].clone()
        } else {
            let initial_prop = task.get_initial_propositional_state_values();
            let mut found = None;
            for (name, &val) in loc_values.iter() {
                if initial_prop.get(loc_var_id).copied() == Some(val) {
                    found = Some(name.clone());
                    break;
                }
            }
            found.ok_or_else(|| -> HeuristicBuildError {
                "missing depot loc fact".to_string().into()
            })?
        };

        let depot = place_idx
            .get(&depot_name)
            .copied()
            .ok_or_else(|| -> HeuristicBuildError {
                "depot not among loc places".to_string().into()
            })?;

        let mut loc_value_for_place = vec![0usize; n_places];
        for (name, &val) in loc_values.iter() {
            if let Some(&idx) = place_idx.get(name) {
                loc_value_for_place[idx] = val;
            }
        }

        let mut customer_infos: Vec<CustomerInfo> = Vec::new();
        for customer_name in &customer_names {
            if let Some(&(var_id, true_value)) = visited_infos.get(customer_name) {
                if let Some(&p_idx) = place_idx.get(customer_name.as_str()) {
                    customer_infos.push(CustomerInfo {
                        place_idx: p_idx,
                        var_id,
                        true_value,
                    });
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;
        let mut demand_by_place = vec![0.0; n_places];
        let mut load_var: Option<usize> = None;
        let mut vehicles_var: Option<usize> = None;

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let full = var.name().trim().to_string();
            let val = initial_numeric.get(id).copied().unwrap_or(0.0);

            if full == "load()" || full == "PNE load()" {
                load_var = Some(id);
            } else if full == "vehicles()" || full == "PNE vehicles()" {
                vehicles_var = Some(id);
            } else if full == "capacity()" || full == "PNE capacity()" {
                capacity = val;
            } else if full == "max_vehicles()" || full == "PNE max_vehicles()" {
                max_vehicles = val;
            } else {
                for customer_name in &customer_names {
                    let expected = format!("demand({})", customer_name);
                    let expected_pne = format!("PNE demand({})", customer_name);
                    if full == expected || full == expected_pne {
                        if let Some(&p_idx) = place_idx.get(customer_name.as_str()) {
                            demand_by_place[p_idx] = val;
                        }
                        break;
                    }
                }
            }
        }

        let mut direct = vec![vec![f64::INFINITY; n_places]; n_places];
        for i in 0..n_places {
            direct[i][i] = 0.0;
        }

        for op in task.get_operators().iter() {
            let cost = metric_operator_cost_from_initial_values(task, op) as f64;
            let op_name = op.name();
            let tokens: Vec<&str> = op_name.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            match tokens[0] {
                "visit" if tokens.len() >= 3 => {
                    let from = tokens[1];
                    let to = tokens[2];
                    if let (Some(&i), Some(&j)) = (place_idx.get(from), place_idx.get(to)) {
                        direct[i][j] = cost;
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    let from = tokens[1];
                    if let Some(&i) = place_idx.get(from) {
                        direct[i][depot] = cost;
                    }
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    let from = tokens[1];
                    let to = tokens[2];
                    if let (Some(&i), Some(&j)) = (place_idx.get(from), place_idx.get(to)) {
                        if direct[i][depot].is_infinite() && direct[depot][j].is_finite() {
                            let est = cost - direct[depot][j];
                            if est >= 0.0 {
                                direct[i][depot] = est;
                            }
                        }
                        if direct[depot][j].is_infinite() && direct[i][depot].is_finite() {
                            let est = cost - direct[i][depot];
                            if est >= 0.0 {
                                direct[depot][j] = est;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let mut shortest = direct.clone();
        for k in 0..n_places {
            for i in 0..n_places {
                if shortest[i][k].is_finite() {
                    for j in 0..n_places {
                        if shortest[k][j].is_finite() {
                            let nd = shortest[i][k] + shortest[k][j];
                            if nd < shortest[i][j] {
                                shortest[i][j] = nd;
                            }
                        }
                    }
                }
            }
        }

        for i in 0..n_places {
            for j in 0..n_places {
                if direct[i][j].is_infinite() {
                    direct[i][j] = shortest[i][j];
                }
            }
        }

        let mut start_cost = vec![0.0; n_places];
        let mut return_cost = vec![0.0; n_places];
        for i in 0..n_places {
            start_cost[i] = shortest[depot][i];
            return_cost[i] = shortest[i][depot];
        }

        Ok(Self {
            depot,
            loc_var_id,
            loc_value_for_place,
            customer_infos,
            direct,
            start_cost,
            return_cost,
            demand_by_place,
            capacity,
            max_vehicles,
            load_var,
            vehicles_var,
            n_prop: task.variables().len(),
            n_num: task.numeric_variables().len(),
        })
    }

    fn find_root(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != cur {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    fn penalty_remaining(&self, remaining: &[usize]) -> f64 {
        let mut penalty = 0.0;
        for &q in remaining {
            if self.start_cost[q].is_finite() {
                penalty += self.start_cost[q];
            }
            if self.return_cost[q].is_finite() {
                penalty += self.return_cost[q];
            }
            penalty += 1_000_000.0;
        }
        penalty
    }

    fn nn_estimate(
        &self,
        current: usize,
        initial_load: f64,
        vehicles_used: f64,
        unvisited: &[usize],
    ) -> f64 {
        let mut remaining: Vec<usize> = unvisited.to_vec();
        if remaining.is_empty() {
            return if current == self.depot {
                0.0
            } else {
                self.return_cost[current]
            };
        }

        let mut new_routes_left = if self.max_vehicles.is_finite() {
            (self.max_vehicles - vehicles_used).floor().max(0.0) as usize
        } else {
            remaining.len().max(1)
        };

        let mut total = 0.0;
        let mut active = current != self.depot;
        let mut last = current;
        let mut current_load = initial_load;
        let mut first_depot_start = current == self.depot;

        while !remaining.is_empty() {
            if !active {
                if first_depot_start {
                    if self.max_vehicles.is_finite() && vehicles_used > self.max_vehicles {
                        total += self.penalty_remaining(&remaining);
                        break;
                    }
                    first_depot_start = false;
                } else {
                    if new_routes_left == 0 {
                        total += self.penalty_remaining(&remaining);
                        break;
                    }
                    new_routes_left -= 1;
                }

                let mut best_q: Option<usize> = None;
                let mut best_start = f64::INFINITY;
                for &q in &remaining {
                    let c = self.start_cost[q];
                    if c < best_start {
                        best_start = c;
                        best_q = Some(q);
                    }
                }

                let q = match best_q {
                    Some(q) => q,
                    None => {
                        total += self.penalty_remaining(&remaining);
                        break;
                    }
                };

                total += self.start_cost[q];
                last = q;
                current_load = self.demand_by_place[q];
                active = true;

                if let Some(pos) = remaining.iter().position(|&x| x == q) {
                    remaining.swap_remove(pos);
                }
            } else {
                let mut best_q: Option<usize> = None;
                let mut best_cost = f64::INFINITY;

                for &q in &remaining {
                    let demand = self.demand_by_place[q];
                    if self.capacity.is_finite()
                        && current_load + demand > self.capacity + 1e-9
                    {
                        continue;
                    }

                    let c = self.direct[last][q];
                    if c.is_finite() && c < best_cost {
                        best_cost = c;
                        best_q = Some(q);
                    }
                }

                if let Some(q) = best_q {
                    total += best_cost;
                    last = q;
                    current_load += self.demand_by_place[q];

                    if let Some(pos) = remaining.iter().position(|&x| x == q) {
                        remaining.swap_remove(pos);
                    }
                } else {
                    total += self.return_cost[last];
                    active = false;
                }
            }
        }

        if active {
            total += self.return_cost[last];
        }

        total.max(0.0)
    }

    fn cw_estimate(&self, unvisited: &[usize], route_limit: usize) -> f64 {
        let m = unvisited.len();
        if m == 0 {
            return 0.0;
        }
        if route_limit == 0 {
            return f64::INFINITY;
        }

        let mut total = 0.0;
        let mut routes: Vec<CwRoute> = Vec::with_capacity(m);
        for (local_idx, &place_idx) in unvisited.iter().enumerate() {
            total += self.start_cost[place_idx] + self.return_cost[place_idx];
            routes.push(CwRoute {
                first: local_idx,
                last: local_idx,
                load: self.demand_by_place[place_idx],
                count: 1,
            });
        }

        if !total.is_finite() {
            return f64::INFINITY;
        }

        let mut parent: Vec<usize> = (0..m).collect();
        let mut route_count = m;

        let mut savings: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..m {
            for j in 0..m {
                if i == j {
                    continue;
                }
                let from = unvisited[i];
                let to = unvisited[j];
                let direct_cost = self.direct[from][to];
                if !direct_cost.is_finite() {
                    continue;
                }
                let saving = self.return_cost[from] + self.start_cost[to] - direct_cost;
                if saving.is_finite() {
                    savings.push((saving, i, j));
                }
            }
        }

        savings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        for (saving, i, j) in savings {
            if route_count <= route_limit && saving <= 0.0 {
                continue;
            }

            let ri = Self::find_root(&mut parent, i);
            let rj = Self::find_root(&mut parent, j);
            if ri == rj {
                continue;
            }

            let ri_data = routes[ri];
            let rj_data = routes[rj];

            let can_merge = ri_data.last == i
                && rj_data.first == j
                && (if self.capacity.is_finite() {
                    ri_data.load + rj_data.load <= self.capacity + 1e-9
                } else {
                    true
                });

            if can_merge {
                parent[ri] = rj;
                routes[rj] = CwRoute {
                    first: ri_data.first,
                    last: rj_data.last,
                    load: ri_data.load + rj_data.load,
                    count: ri_data.count + rj_data.count,
                };
                total -= saving;
                route_count -= 1;
            }
        }

        if route_count > route_limit {
            let excess = (route_count - route_limit) as f64;
            total += excess * 1_000_000.0;
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

        let mut numeric_values = vec![0.0; self.n_num];
        registry
            .view(state)
            .fill_numeric(&mut numeric_values)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

        let mut propositional_values = vec![0usize; self.n_prop];
        let _ = state.fill_state(registry, &mut propositional_values);

        let load = self
            .load_var
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(0.0);

        let vehicles = self
            .vehicles_var
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(1.0);

        let mut current_idx = self.depot;
        for (idx, &val) in self.loc_value_for_place.iter().enumerate() {
            if propositional_values[self.loc_var_id] == val {
                current_idx = idx;
                break;
            }
        }

        let mut unvisited: Vec<usize> = Vec::new();
        for info in &self.customer_infos {
            if propositional_values[info.var_id] != info.true_value {
                unvisited.push(info.place_idx);
            }
        }

        if unvisited.is_empty() {
            return if current_idx != self.depot {
                Ok(self.return_cost[current_idx].max(0.0))
            } else {
                Ok(0.0)
            };
        }

        let h1 = self.nn_estimate(current_idx, load, vehicles, &unvisited);

        let cw_route_limit = if current_idx != self.depot {
            if self.max_vehicles.is_finite() {
                (self.max_vehicles - vehicles).floor().max(0.0) as usize
            } else {
                unvisited.len()
            }
        } else {
            if self.max_vehicles.is_finite() {
                (self.max_vehicles - vehicles + 1.0).floor().max(0.0) as usize
            } else {
                unvisited.len()
            }
        };

        let cw_cost = self.cw_estimate(&unvisited, cw_route_limit);

        let mut h = h1;
        if cw_cost.is_finite() {
            let h2 = if current_idx != self.depot {
                self.return_cost[current_idx] + cw_cost
            } else {
                cw_cost
            };
            if h2 < h {
                h = h2;
            }
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
