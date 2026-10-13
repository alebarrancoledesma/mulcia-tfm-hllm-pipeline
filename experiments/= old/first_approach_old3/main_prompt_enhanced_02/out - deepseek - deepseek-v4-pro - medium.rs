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

const INF_DIST: f64 = 1e30;
const LARGE_PENALTY: f64 = 1e18;

fn parse_paren(s: &str, prefix: &str) -> Option<String> {
    let start = s.find(prefix)? + prefix.len();
    let rest = &s[start..];
    let end = rest.find(')')?;
    Some(rest[..end].to_string())
}

fn fits_direct(load: f64, demand: f64, capacity: f64) -> bool {
    if capacity > 1e300 {
        return true;
    }
    load + demand <= capacity + 1e-9
}

fn fits_new_vehicle(demand: f64, capacity: f64) -> bool {
    if capacity > 1e300 {
        return true;
    }
    demand <= capacity + 1e-9
}

fn greedy_completion(
    start_place: usize,
    start_load: f64,
    start_vehicles: f64,
    unvisited: &[usize],
    flags: &mut [bool],
    depot_idx: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: &[f64],
    dist: &[Vec<f64>],
    via_depot: &[Vec<f64>],
    nearest: &[Vec<usize>],
) -> f64 {
    let mut cur = start_place;
    let mut cur_load = start_load;
    let mut vehicles_used = start_vehicles;
    let mut remaining = 0usize;
    for &c in unvisited {
        if flags[c] {
            remaining += 1;
        }
    }

    let mut total = 0.0;
    while remaining > 0 {
        let mut next_direct: Option<usize> = None;
        for &cand in &nearest[cur] {
            if flags[cand] && fits_direct(cur_load, demands[cand], capacity) {
                next_direct = Some(cand);
                break;
            }
        }

        if let Some(c) = next_direct {
            let d = dist[cur][c];
            if d >= INF_DIST / 2.0 {
                return LARGE_PENALTY;
            }
            total += d;
            cur = c;
            cur_load += demands[c];
            flags[c] = false;
            remaining -= 1;
        } else {
            if vehicles_used + 1.0 > max_vehicles + 1e-9 {
                return LARGE_PENALTY;
            }

            let mut next_route: Option<usize> = None;
            for &cand in &nearest[depot_idx] {
                if flags[cand] && fits_new_vehicle(demands[cand], capacity) {
                    next_route = Some(cand);
                    break;
                }
            }

            let c = match next_route {
                Some(c) => c,
                None => return LARGE_PENALTY,
            };

            let vcost = if via_depot[cur][c] < INF_DIST / 2.0 {
                via_depot[cur][c]
            } else if dist[cur][depot_idx] < INF_DIST / 2.0
                && dist[depot_idx][c] < INF_DIST / 2.0
            {
                dist[cur][depot_idx] + dist[depot_idx][c]
            } else {
                return LARGE_PENALTY;
            };

            total += vcost;
            vehicles_used += 1.0;
            cur = c;
            cur_load = demands[c];
            flags[c] = false;
            remaining -= 1;
        }
    }

    if cur != depot_idx {
        let d = dist[cur][depot_idx];
        if d < INF_DIST / 2.0 {
            total += d;
        } else {
            return LARGE_PENALTY;
        }
    }

    total
}

struct LocFact {
    name: String,
    var_id: usize,
    value: usize,
}

struct CustomerInfo {
    name: String,
    place_idx: usize,
    visited_var: usize,
    visited_value: usize,
    demand: f64,
}

struct LlmHeuristic {
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    max_vehicles_id: Option<usize>,
    capacity_id: Option<usize>,
    capacity: f64,
    max_vehicles: f64,
    depot_idx: usize,
    place_names: Vec<String>,
    place_index: HashMap<String, usize>,
    customers: Vec<CustomerInfo>,
    loc_facts: Vec<LocFact>,
    dist: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    demands: Vec<f64>,
    nearest_from: Vec<Vec<usize>>,
    num_numeric_vars: usize,
    num_prop_vars: usize,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_prop: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<usize>>,
    scratch_visited_flags: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let num_numeric_vars = task.numeric_variables().len();
        let num_prop_vars = task.variables().len();
        let initial_numeric = task.get_initial_numeric_state_values().to_vec();

        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut max_vehicles_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut demand_by_name: HashMap<String, f64> = HashMap::new();
        let mut fallback_costs: Vec<(String, String, f64)> = Vec::new();

        for (id, nvar) in task.numeric_variables().iter().enumerate() {
            let name = nvar.name();
            if name.contains("max_vehicles()") {
                max_vehicles_id = Some(id);
            }
            if name.contains("vehicles()") && !name.contains("max_vehicles()") {
                vehicles_id = Some(id);
            }
            if name.contains("load()") {
                load_id = Some(id);
            }
            if name.contains("capacity()") {
                capacity_id = Some(id);
            }
            if let Some(obj) = parse_paren(name, "demand(") {
                let obj = obj.trim().to_string();
                if id < initial_numeric.len() {
                    demand_by_name.insert(obj, initial_numeric[id]);
                }
            }
            if let Some(inside) = parse_paren(name, "travel-cost(") {
                let parts: Vec<&str> = inside.split_whitespace().collect();
                if parts.len() == 2 && id < initial_numeric.len() {
                    fallback_costs.push((
                        parts[0].to_string(),
                        parts[1].to_string(),
                        initial_numeric[id],
                    ));
                }
            }
        }

        let mut loc_facts: Vec<LocFact> = Vec::new();
        let mut loc_objects: HashSet<String> = HashSet::new();
        let mut visited_objects: HashSet<String> = HashSet::new();
        let mut visited_facts: HashMap<String, (usize, usize)> = HashMap::new();

        for var_id in 0..num_prop_vars {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = format!("{:?}", task.get_fact_name(&fact));

                if fact_name.contains("Atom loc(") {
                    if let Some(obj) = parse_paren(&fact_name, "loc(") {
                        loc_objects.insert(obj.clone());
                        loc_facts.push(LocFact {
                            name: obj,
                            var_id,
                            value,
                        });
                    }
                }

                if fact_name.contains("Atom visited(") {
                    if let Some(obj) = parse_paren(&fact_name, "visited(") {
                        visited_objects.insert(obj.clone());
                        visited_facts.insert(obj, (var_id, value));
                    }
                }
            }
        }

        let mut depot_name: Option<String> = None;
        let mut loc_names: Vec<String> = loc_objects.iter().cloned().collect();
        loc_names.sort();
        for obj in loc_names {
            if !visited_objects.contains(&obj) {
                depot_name = Some(obj);
                break;
            }
        }
        let depot_name = match depot_name {
            Some(name) => name,
            None => {
                return Err(
                    "Could not identify depot from propositional loc/visited facts."
                        .to_string()
                        .into(),
                );
            }
        };

        let mut customer_names: Vec<String> = visited_objects.iter().cloned().collect();
        customer_names.sort();

        let mut place_names = Vec::new();
        place_names.push(depot_name.clone());
        for c in &customer_names {
            place_names.push(c.clone());
        }

        let mut place_index: HashMap<String, usize> = HashMap::new();
        for (idx, name) in place_names.iter().enumerate() {
            place_index.insert(name.clone(), idx);
        }

        let depot_idx = place_index[&depot_name];
        let n_places = place_names.len();

        let capacity = capacity_id
            .map(|id| initial_numeric[id].max(0.0))
            .unwrap_or(f64::MAX);
        let max_vehicles = max_vehicles_id
            .map(|id| initial_numeric[id].max(0.0))
            .unwrap_or(f64::MAX);

        let mut demands = vec![0.0; n_places];
        let mut customers: Vec<CustomerInfo> = Vec::new();
        for cname in &customer_names {
            let place_idx = place_index[cname];
            let (visited_var, visited_value) = visited_facts[cname];
            let demand = demand_by_name.get(cname).copied().unwrap_or(0.0);
            demands[place_idx] = demand;
            customers.push(CustomerInfo {
                name: cname.clone(),
                place_idx,
                visited_var,
                visited_value,
                demand,
            });
        }

        let mut dist = vec![vec![INF_DIST; n_places]; n_places];
        for i in 0..n_places {
            dist[i][i] = 0.0;
        }
        let mut via_depot_cost = vec![vec![INF_DIST; n_places]; n_places];
        for i in 0..n_places {
            via_depot_cost[i][i] = 0.0;
        }

        for op in task.get_operators().iter() {
            let name = op.name();
            let tokens: Vec<&str> = name.split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, op);

            if tokens.len() == 3 && tokens[0] == "visit" {
                if let (Some(&a), Some(&b)) =
                    (place_index.get(tokens[1]), place_index.get(tokens[2]))
                {
                    dist[a][b] = cost;
                }
            } else if tokens.len() == 3 && tokens[0] == "visit-via-depot" {
                if let (Some(&a), Some(&b)) =
                    (place_index.get(tokens[1]), place_index.get(tokens[2]))
                {
                    via_depot_cost[a][b] = cost;
                }
            } else if tokens.len() == 2 && tokens[0] == "return-to-depot" {
                if let Some(&a) = place_index.get(tokens[1]) {
                    dist[a][depot_idx] = cost;
                }
            }
        }

        for (a_name, b_name, cost) in fallback_costs {
            if let (Some(&a), Some(&b)) =
                (place_index.get(&a_name), place_index.get(&b_name))
            {
                if dist[a][b] >= INF_DIST / 2.0 {
                    dist[a][b] = cost;
                }
            }
        }

        for a in 0..n_places {
            for b in 0..n_places {
                if a != b
                    && dist[a][b] >= INF_DIST / 2.0
                    && dist[a][depot_idx] < INF_DIST / 2.0
                    && dist[depot_idx][b] < INF_DIST / 2.0
                {
                    dist[a][b] = dist[a][depot_idx] + dist[depot_idx][b];
                }
                if a != b
                    && via_depot_cost[a][b] >= INF_DIST / 2.0
                    && dist[a][depot_idx] < INF_DIST / 2.0
                    && dist[depot_idx][b] < INF_DIST / 2.0
                {
                    via_depot_cost[a][b] = dist[a][depot_idx] + dist[depot_idx][b];
                }
            }
        }

        let depot_ok_out = dist[depot_idx].iter().any(|&v| v < INF_DIST / 2.0);
        let depot_ok_in = (0..n_places)
            .any(|a| a != depot_idx && dist[a][depot_idx] < INF_DIST / 2.0);
        if !depot_ok_out && !depot_ok_in {
            return Err(
                "Could not reconstruct any usable travel costs from grounded operators or PNE fallback."
                    .to_string()
                    .into(),
            );
        }

        let mut nearest_from: Vec<Vec<usize>> = vec![Vec::new(); n_places];
        for a in 0..n_places {
            let mut list: Vec<usize> = (0..n_places)
                .filter(|&b| a != b && dist[a][b] < INF_DIST / 2.0)
                .collect();
            list.sort_by(|&x, &y| {
                dist[a][x]
                    .partial_cmp(&dist[a][y])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_from[a] = list;
        }

        Ok(Self {
            load_id,
            vehicles_id,
            max_vehicles_id,
            capacity_id,
            capacity,
            max_vehicles,
            depot_idx,
            place_names,
            place_index,
            customers,
            loc_facts,
            dist,
            via_depot_cost,
            demands,
            nearest_from,
            num_numeric_vars,
            num_prop_vars,
            scratch_numeric: RefCell::new(vec![0.0; num_numeric_vars]),
            scratch_prop: RefCell::new(vec![0; num_prop_vars]),
            scratch_unvisited: RefCell::new(Vec::new()),
            scratch_visited_flags: RefCell::new(vec![false; n_places]),
        })
    }

    fn average_depot_arc_cost(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0.0;
        for cust in &self.customers {
            let a = self.depot_idx;
            let b = cust.place_idx;
            if self.dist[a][b] < INF_DIST / 2.0 {
                sum += self.dist[a][b];
                count += 1.0;
            }
            if self.dist[b][a] < INF_DIST / 2.0 {
                sum += self.dist[b][a];
                count += 1.0;
            }
        }
        if count > 0.0 {
            sum / count
        } else {
            0.0
        }
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

        let (load, vehicles_used) = {
            let mut numeric_values = self.scratch_numeric.borrow_mut();
            numeric_values.resize(self.num_numeric_vars, 0.0);
            registry
                .view(state)
                .fill_numeric(&mut *numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
            let load = self
                .load_id
                .map(|id| numeric_values[id])
                .unwrap_or(0.0)
                .max(0.0);
            let vehicles_used = self
                .vehicles_id
                .map(|id| numeric_values[id])
                .unwrap_or(1.0)
                .max(0.0);
            (load, vehicles_used)
        };

        let mut prop_values = self.scratch_prop.borrow_mut();
        prop_values.resize(self.num_prop_vars, 0);
        state
            .fill_state(registry, &mut *prop_values)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;

        let mut cur_place = self.depot_idx;
        for loc in &self.loc_facts {
            if prop_values[loc.var_id] == loc.value {
                cur_place = self
                    .place_index
                    .get(&loc.name)
                    .copied()
                    .unwrap_or(self.depot_idx);
                break;
            }
        }

        let mut unvisited = self.scratch_unvisited.borrow_mut();
        unvisited.clear();
        let mut remaining_demand = 0.0;
        for cust in &self.customers {
            let visited = prop_values[cust.visited_var] == cust.visited_value;
            if !visited {
                unvisited.push(cust.place_idx);
                remaining_demand += self.demands[cust.place_idx];
            }
        }
        drop(prop_values);

        if unvisited.is_empty() {
            let h = if cur_place != self.depot_idx {
                let d = self.dist[cur_place][self.depot_idx];
                if d < INF_DIST / 2.0 {
                    d
                } else {
                    return Ok(LARGE_PENALTY);
                }
            } else {
                0.0
            };
            return Ok(h);
        }

        let avail_vehicles = (self.max_vehicles - vehicles_used).max(0.0);
        let available_capacity = (self.capacity - load).max(0.0)
            + if avail_vehicles >= f64::MAX / self.capacity.max(1.0) {
                f64::MAX
            } else {
                avail_vehicles * self.capacity
            };

        if remaining_demand > available_capacity + 1e-6 {
            return Ok(LARGE_PENALTY);
        }

        let mut flags = self.scratch_visited_flags.borrow_mut();
        flags.clear();
        flags.resize(self.place_names.len(), false);
        for &c in unvisited.iter() {
            flags[c] = true;
        }

        let h_routing = greedy_completion(
            cur_place,
            load,
            vehicles_used,
            unvisited.as_slice(),
            flags.as_mut_slice(),
            self.depot_idx,
            self.capacity,
            self.max_vehicles,
            &self.demands,
            &self.dist,
            &self.via_depot_cost,
            &self.nearest_from,
        );

        let h_capacity = if remaining_demand > available_capacity + 1e-6 {
            LARGE_PENALTY
        } else {
            0.0
        };

        let avg_depot = self.average_depot_arc_cost();
        let needed_routes = (remaining_demand / self.capacity.max(1.0)).ceil();
        let remaining_routes = (self.max_vehicles - vehicles_used).max(0.0).floor();
        let h_vehicle_pressure = if self.max_vehicles < f64::MAX && !unvisited.is_empty() {
            (needed_routes / remaining_routes.max(1.0)) * avg_depot * 0.05
        } else {
            0.0
        };

        if h_routing >= LARGE_PENALTY / 2.0 || h_capacity >= LARGE_PENALTY / 2.0 {
            return Ok(LARGE_PENALTY);
        }

        Ok(h_routing + h_capacity + h_vehicle_pressure)
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
