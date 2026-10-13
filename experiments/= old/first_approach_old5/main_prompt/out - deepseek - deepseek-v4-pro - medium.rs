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

const EPS: f64 = 1e-6;

fn parse_fact_name(name: &str) -> Option<(String, Vec<String>)> {
    let open = name.find('(')?;
    let close = name.rfind(')')?;
    let before = &name[..open];
    let predicate = before.split_whitespace().last()?.to_string();
    let inner = &name[open + 1..close];
    let args = inner
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Some((predicate, args))
}

fn numeric_id_by_name(task: &dyn AbstractNumericTask, target: &str) -> Option<usize> {
    task.numeric_variables()
        .iter()
        .enumerate()
        .find_map(|(idx, var)| {
            let name = var.name().trim();
            if name == target {
                Some(idx)
            } else {
                name.strip_prefix("PNE ")
                    .and_then(|s| if s == target { Some(idx) } else { None })
            }
        })
}

struct LlmHeuristic {
    depot: usize,
    customers: Vec<usize>,
    visited_var: Option<usize>,
    visited_value: Vec<usize>,
    loc_var: Option<usize>,
    loc_value: Vec<usize>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    capacity: f64,
    max_vehicles: f64,
    demand: Vec<f64>,
    dist: Vec<Vec<f64>>,
    neighbors: Vec<Vec<usize>>,
    depot_neighbors: Vec<usize>,
    penalty: f64,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut places: Vec<String> = Vec::new();
        let mut place_to_idx: HashMap<String, usize> = HashMap::new();
        let mut loc_var: Option<usize> = None;
        let mut loc_value_by_obj: HashMap<String, usize> = HashMap::new();
        let mut visited_var: Option<usize> = None;
        let mut visited_value_by_obj: HashMap<String, usize> = HashMap::new();
        let mut visited_objects: HashSet<String> = HashSet::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);
                if let Some((pred, args)) = parse_fact_name(&fact_name) {
                    if pred == "loc" && args.len() == 1 {
                        let obj = &args[0];
                        if !place_to_idx.contains_key(obj) {
                            let idx = places.len();
                            places.push(obj.clone());
                            place_to_idx.insert(obj.clone(), idx);
                        }
                        loc_var = Some(var_id);
                        loc_value_by_obj.insert(obj.clone(), value);
                    } else if pred == "visited" && args.len() == 1 {
                        let obj = &args[0];
                        if !place_to_idx.contains_key(obj) {
                            let idx = places.len();
                            places.push(obj.clone());
                            place_to_idx.insert(obj.clone(), idx);
                        }
                        visited_objects.insert(obj.clone());
                        visited_var = Some(var_id);
                        visited_value_by_obj.insert(obj.clone(), value);
                    }
                }
            }
        }

        let num_places = places.len();
        if num_places == 0 {
            return Ok(Self {
                depot: 0,
                customers: Vec::new(),
                visited_var,
                visited_value: Vec::new(),
                loc_var,
                loc_value: Vec::new(),
                load_var: None,
                vehicles_var: None,
                capacity: f64::MAX,
                max_vehicles: f64::MAX,
                demand: Vec::new(),
                dist: Vec::new(),
                neighbors: Vec::new(),
                depot_neighbors: Vec::new(),
                penalty: 0.0,
            });
        }

        let mut loc_value = vec![0usize; num_places];
        let mut visited_value = vec![0usize; num_places];
        for (obj, &idx) in &place_to_idx {
            if let Some(&v) = loc_value_by_obj.get(obj) {
                loc_value[idx] = v;
            }
            if let Some(&v) = visited_value_by_obj.get(obj) {
                visited_value[idx] = v;
            }
        }

        let mut depot = 0usize;
        for (obj, &idx) in &place_to_idx {
            if !visited_objects.contains(obj) {
                depot = idx;
                break;
            }
        }

        let mut customers = Vec::new();
        for (obj, &idx) in &place_to_idx {
            if visited_objects.contains(obj) {
                customers.push(idx);
            }
        }

        let mut dist = vec![vec![f64::INFINITY; num_places]; num_places];
        for i in 0..num_places {
            dist[i][i] = 0.0;
        }

        let mut via_depot_pairs: Vec<(usize, usize, f64)> = Vec::new();
        let operators = task.get_operators();
        for op in operators.iter() {
            let cost = metric_operator_cost_from_initial_values(task, op);
            let name = op.name();
            let tokens: Vec<&str> = name.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            match tokens[0] {
                "visit" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_to_idx.get(tokens[1]), place_to_idx.get(tokens[2]))
                    {
                        dist[from][to] = cost;
                    }
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_to_idx.get(tokens[1]), place_to_idx.get(tokens[2]))
                    {
                        via_depot_pairs.push((from, to, cost));
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = place_to_idx.get(tokens[1]) {
                        dist[from][depot] = cost;
                    }
                }
                _ => {}
            }
        }

        for &(from, to, cost) in &via_depot_pairs {
            if !dist[from][depot].is_finite() && dist[depot][to].is_finite() {
                dist[from][depot] = cost - dist[depot][to];
            }
            if !dist[depot][to].is_finite() && dist[from][depot].is_finite() {
                dist[depot][to] = cost - dist[from][depot];
            }
        }

        let mut demand = Vec::with_capacity(customers.len());
        for &p in &customers {
            let obj = &places[p];
            let target = format!("demand({})", obj);
            if let Some(id) = numeric_id_by_name(task, &target) {
                demand.push(task.get_initial_numeric_state_values()[id]);
            } else {
                demand.push(0.0);
            }
        }

        let capacity = numeric_id_by_name(task, "capacity()")
            .map(|id| task.get_initial_numeric_state_values()[id])
            .unwrap_or(f64::MAX);
        let max_vehicles = numeric_id_by_name(task, "max_vehicles()")
            .map(|id| task.get_initial_numeric_state_values()[id])
            .unwrap_or(f64::MAX);
        let load_var = numeric_id_by_name(task, "load()");
        let vehicles_var = numeric_id_by_name(task, "vehicles()");

        let mut neighbors = Vec::with_capacity(num_places);
        for p in 0..num_places {
            let mut list: Vec<usize> = (0..customers.len()).collect();
            list.sort_by(|&a, &b| {
                let da = dist[p][customers[a]];
                let db = dist[p][customers[b]];
                if da.is_finite() && db.is_finite() {
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                } else if da.is_finite() {
                    Ordering::Less
                } else if db.is_finite() {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            neighbors.push(list);
        }
        let depot_neighbors = neighbors[depot].clone();

        let mut max_distance = 0.0f64;
        for row in &dist {
            for &d in row {
                if d.is_finite() && d > max_distance {
                    max_distance = d;
                }
            }
        }
        let penalty = if max_distance > 0.0 {
            max_distance * 10.0 + 1.0
        } else {
            1.0
        };

        Ok(Self {
            depot,
            customers,
            visited_var,
            visited_value,
            loc_var,
            loc_value,
            load_var,
            vehicles_var,
            capacity,
            max_vehicles,
            demand,
            dist,
            neighbors,
            depot_neighbors,
            penalty,
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

        if self.customers.is_empty() {
            return Ok(0.0);
        }

        let mut prop_values = vec![0usize; task.variables().len()];
        state.fill_state(registry, &mut prop_values);

        let mut numeric_values = vec![0.0f64; task.numeric_variables().len()];
        registry
            .view(state)
            .fill_numeric(&mut numeric_values)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

        let mut cur = self.depot;
        if let Some(loc_var) = self.loc_var {
            if loc_var < prop_values.len() {
                for p in 0..self.loc_value.len() {
                    if prop_values[loc_var] == self.loc_value[p] {
                        cur = p;
                        break;
                    }
                }
            }
        }

        let mut unvisited = vec![false; self.customers.len()];
        let mut remaining = 0usize;
        if let Some(visited_var) = self.visited_var {
            if visited_var < prop_values.len() {
                for ci in 0..self.customers.len() {
                    let p = self.customers[ci];
                    if prop_values[visited_var] != self.visited_value[p] {
                        unvisited[ci] = true;
                        remaining += 1;
                    }
                }
            } else {
                for ci in 0..self.customers.len() {
                    unvisited[ci] = true;
                    remaining += 1;
                }
            }
        } else {
            for ci in 0..self.customers.len() {
                unvisited[ci] = true;
                remaining += 1;
            }
        }

        if remaining == 0 {
            if cur != self.depot {
                let ret = self.dist[cur][self.depot];
                return Ok(if ret.is_finite() { ret } else { self.penalty });
            }
            return Ok(0.0);
        }

        let mut total = 0.0f64;
        let mut cur_load = self
            .load_var
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(0.0);
        let mut used = self
            .vehicles_var
            .and_then(|id| numeric_values.get(id).copied())
            .unwrap_or(1.0);

        while remaining > 0 {
            if cur == self.depot {
                let mut chosen = usize::MAX;
                let mut chosen_cost = f64::INFINITY;
                for &ci in &self.depot_neighbors {
                    if !unvisited[ci] {
                        continue;
                    }
                    if self.demand[ci] <= self.capacity + EPS {
                        let cost = self.dist[self.depot][self.customers[ci]];
                        if cost.is_finite() && cost < chosen_cost {
                            chosen_cost = cost;
                            chosen = ci;
                        }
                    }
                }

                if chosen == usize::MAX {
                    for &ci in &self.depot_neighbors {
                        if !unvisited[ci] {
                            continue;
                        }
                        let cost = self.dist[self.depot][self.customers[ci]];
                        if cost.is_finite() && cost < chosen_cost {
                            chosen_cost = cost;
                            chosen = ci;
                        }
                    }
                }

                if chosen == usize::MAX {
                    total += self.penalty * remaining as f64;
                    break;
                }

                total += chosen_cost;
                cur = self.customers[chosen];
                cur_load = self.demand[chosen];
                unvisited[chosen] = false;
                remaining -= 1;
            } else {
                let mut chosen_cont = usize::MAX;
                let mut cont_cost = f64::INFINITY;
                for &ci in &self.neighbors[cur] {
                    if !unvisited[ci] {
                        continue;
                    }
                    if cur_load + self.demand[ci] <= self.capacity + EPS {
                        let cost = self.dist[cur][self.customers[ci]];
                        if cost.is_finite() && cost < cont_cost {
                            cont_cost = cost;
                            chosen_cont = ci;
                        }
                    }
                }

                let mut chosen_new = usize::MAX;
                let mut new_cost = f64::INFINITY;
                if used + 1.0 <= self.max_vehicles + EPS {
                    let ret = self.dist[cur][self.depot];
                    if ret.is_finite() {
                        for &ci in &self.depot_neighbors {
                            if !unvisited[ci] {
                                continue;
                            }
                            if self.demand[ci] <= self.capacity + EPS {
                                let out = self.dist[self.depot][self.customers[ci]];
                                if out.is_finite() {
                                    let route_cost = ret + out;
                                    if route_cost < new_cost {
                                        new_cost = route_cost;
                                        chosen_new = ci;
                                    }
                                }
                            }
                        }
                    }
                }

                if chosen_cont == usize::MAX && chosen_new == usize::MAX {
                    for &ci in &self.neighbors[cur] {
                        if !unvisited[ci] {
                            continue;
                        }
                        let cost = self.dist[cur][self.customers[ci]];
                        if cost.is_finite() && cost < cont_cost {
                            cont_cost = cost;
                            chosen_cont = ci;
                        }
                    }

                    if chosen_cont != usize::MAX {
                        total += cont_cost;
                        cur = self.customers[chosen_cont];
                        cur_load += self.demand[chosen_cont];
                        unvisited[chosen_cont] = false;
                        remaining -= 1;
                        continue;
                    }

                    total += self.penalty * remaining as f64;
                    break;
                }

                if chosen_new != usize::MAX
                    && (chosen_cont == usize::MAX || new_cost < cont_cost)
                {
                    total += new_cost;
                    cur = self.customers[chosen_new];
                    cur_load = self.demand[chosen_new];
                    used += 1.0;
                    unvisited[chosen_new] = false;
                    remaining -= 1;
                } else {
                    total += cont_cost;
                    cur = self.customers[chosen_cont];
                    cur_load += self.demand[chosen_cont];
                    unvisited[chosen_cont] = false;
                    remaining -= 1;
                }
            }
        }

        if cur != self.depot {
            let ret = self.dist[cur][self.depot];
            if ret.is_finite() {
                total += ret;
            } else {
                total += self.penalty;
            }
        }

        Ok(total.max(0.0))
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
