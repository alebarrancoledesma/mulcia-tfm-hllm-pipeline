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

const LARGE_PENALTY: f64 = 1e15;

#[derive(Clone, Copy)]
struct FactId {
    var_id: usize,
    value: usize,
}

fn parse_inner<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let start = name.find(prefix)?;
    let after = start + prefix.len();
    let s = &name[after..];
    let end = s.find(')')?;
    Some(&s[..end])
}

fn parse_two_arg_name(name: &str, prefix: &str) -> Option<(String, String)> {
    let inner = parse_inner(name, prefix)?;
    let parts: Vec<&str> = inner
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() == 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

struct LlmHeuristic {
    n_places: usize,
    depot_index: usize,
    customer_indices: Vec<usize>,
    is_customer: Vec<bool>,
    loc_facts: Vec<FactId>,
    visited_facts: Vec<Option<FactId>>,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    demand: Vec<f64>,
    cost: Vec<Vec<f64>>,
    via_cost: Vec<Vec<f64>>,
    nearest_direct: Vec<Vec<(f64, usize)>>,
    nearest_via: Vec<Vec<(f64, usize)>>,
    avg_depot_cost: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_propo: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_map: HashMap<String, FactId> = HashMap::new();
        let mut visited_map: HashMap<String, FactId> = HashMap::new();

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = format!("{}", task.get_fact_name(&fact));
                if let Some(obj) = parse_inner(&fact_name, "loc(") {
                    loc_map.entry(obj.to_string()).or_insert(FactId { var_id, value });
                } else if let Some(obj) = parse_inner(&fact_name, "visited(") {
                    visited_map.entry(obj.to_string()).or_insert(FactId { var_id, value });
                }
            }
        }

        let mut all_names: Vec<String> = loc_map.keys().cloned().collect();
        for name in visited_map.keys() {
            if !all_names.contains(name) {
                all_names.push(name.clone());
            }
        }
        all_names.sort();

        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        for (idx, name) in all_names.iter().enumerate() {
            name_to_idx.insert(name.clone(), idx);
        }

        let depot_name = all_names
            .iter()
            .find(|n| loc_map.contains_key(*n) && !visited_map.contains_key(*n))
            .ok_or_else(|| HeuristicBuildError::from("CVRP heuristic: no depot found".to_string()))?;
        let depot_index = name_to_idx[depot_name.as_str()];

        let n_places = all_names.len();
        let mut is_customer = vec![false; n_places];
        let mut customer_indices = Vec::new();
        for (idx, name) in all_names.iter().enumerate() {
            if visited_map.contains_key(name) {
                is_customer[idx] = true;
                customer_indices.push(idx);
            }
        }

        let mut loc_facts = vec![FactId { var_id: 0, value: 0 }; n_places];
        for (name, idx) in &name_to_idx {
            if let Some(fact) = loc_map.get(name) {
                loc_facts[*idx] = *fact;
            } else {
                return Err(HeuristicBuildError::from(format!(
                    "CVRP heuristic: missing loc fact for {name}"
                )));
            }
        }

        let mut visited_facts: Vec<Option<FactId>> = vec![None; n_places];
        for (name, idx) in &name_to_idx {
            if let Some(fact) = visited_map.get(name) {
                visited_facts[*idx] = Some(*fact);
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity_id = None;
        let mut max_vehicles_id = None;
        let mut demand_by_place = vec![0.0; n_places];

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();

            if load_id.is_none() && name.contains("load()") && !name.contains("total-cost") {
                load_id = Some(id);
            }
            if vehicles_id.is_none()
                && name.contains("vehicles()")
                && !name.contains("max_vehicles()")
            {
                vehicles_id = Some(id);
            }
            if capacity_id.is_none() && name.contains("capacity()") {
                capacity_id = Some(id);
            }
            if max_vehicles_id.is_none() && name.contains("max_vehicles()") {
                max_vehicles_id = Some(id);
            }

            if let Some(obj) = parse_inner(&name, "demand(") {
                let obj = obj.trim().to_string();
                if let Some(&idx) = name_to_idx.get(&obj) {
                    if is_customer[idx] {
                        demand_by_place[idx] = initial_numeric[id];
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            HeuristicBuildError::from("CVRP heuristic: could not find load() numeric variable".to_string())
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            HeuristicBuildError::from("CVRP heuristic: could not find vehicles() numeric variable".to_string())
        })?;

        let capacity = capacity_id.map(|id| initial_numeric[id]).unwrap_or(f64::MAX);
        let max_vehicles = max_vehicles_id
            .map(|id| initial_numeric[id])
            .unwrap_or(f64::MAX);

        let mut cost = vec![vec![f64::INFINITY; n_places]; n_places];
        let mut via_cost = vec![vec![f64::INFINITY; n_places]; n_places];
        cost[depot_index][depot_index] = 0.0;

        let operators = task.get_operators();
        for op in &operators {
            let op_name = op.name().to_string();
            let parts: Vec<&str> = op_name.split_whitespace().collect();
            let op_cost = metric_operator_cost_from_initial_values(task, op);

            if parts.len() == 3 && parts[0] == "visit" {
                if let (Some(&from_idx), Some(&to_idx)) =
                    (name_to_idx.get(parts[1]), name_to_idx.get(parts[2]))
                {
                    cost[from_idx][to_idx] = op_cost;
                }
            } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
                if let (Some(&from_idx), Some(&to_idx)) =
                    (name_to_idx.get(parts[1]), name_to_idx.get(parts[2]))
                {
                    via_cost[from_idx][to_idx] = op_cost;
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&from_idx) = name_to_idx.get(parts[1]) {
                    cost[from_idx][depot_index] = op_cost;
                }
            }
        }

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if let Some((from_name, to_name)) = parse_two_arg_name(&name, "travel-cost-via-depot(") {
                if let (Some(&from_idx), Some(&to_idx)) =
                    (name_to_idx.get(&from_name), name_to_idx.get(&to_name))
                {
                    via_cost[from_idx][to_idx] = initial_numeric[id];
                }
            }
            if let Some((from_name, to_name)) = parse_two_arg_name(&name, "travel-cost(") {
                if let (Some(&from_idx), Some(&to_idx)) =
                    (name_to_idx.get(&from_name), name_to_idx.get(&to_name))
                {
                    cost[from_idx][to_idx] = initial_numeric[id];
                }
            }
        }

        for from in 0..n_places {
            for to in 0..n_places {
                if cost[from][to].is_infinite()
                    && cost[from][depot_index].is_finite()
                    && cost[depot_index][to].is_finite()
                {
                    cost[from][to] = cost[from][depot_index] + cost[depot_index][to];
                }
            }
        }

        for from in 0..n_places {
            if !is_customer[from] {
                continue;
            }
            for to in 0..n_places {
                if is_customer[to]
                    && via_cost[from][to].is_infinite()
                    && cost[from][depot_index].is_finite()
                    && cost[depot_index][to].is_finite()
                {
                    via_cost[from][to] = cost[from][depot_index] + cost[depot_index][to];
                }
            }
        }

        for &c in &customer_indices {
            if cost[depot_index][c].is_infinite() || cost[c][depot_index].is_infinite() {
                return Err(HeuristicBuildError::from(
                    "CVRP heuristic: could not reconstruct essential depot-customer travel costs".to_string(),
                ));
            }
        }

        let mut nearest_direct = Vec::with_capacity(n_places);
        for from in 0..n_places {
            let mut list: Vec<(f64, usize)> = customer_indices
                .iter()
                .filter_map(|&to| {
                    let c = cost[from][to];
                    if c.is_finite() {
                        Some((c, to))
                    } else {
                        None
                    }
                })
                .collect();
            list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            nearest_direct.push(list);
        }

        let mut nearest_via = Vec::with_capacity(n_places);
        for from in 0..n_places {
            if is_customer[from] {
                let mut list: Vec<(f64, usize)> = customer_indices
                    .iter()
                    .filter_map(|&to| {
                        let c = via_cost[from][to];
                        if c.is_finite() {
                            Some((c, to))
                        } else {
                            None
                        }
                    })
                    .collect();
                list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                nearest_via.push(list);
            } else {
                nearest_via.push(Vec::new());
            }
        }

        let mut avg_depot_cost = 0.0;
        let mut cnt = 0.0;
        for &c in &customer_indices {
            let out = cost[depot_index][c];
            let back = cost[c][depot_index];
            if out.is_finite() && back.is_finite() {
                avg_depot_cost += out + back;
                cnt += 1.0;
            }
        }
        if cnt > 0.0 {
            avg_depot_cost /= cnt;
        }

        Ok(Self {
            n_places,
            depot_index,
            customer_indices,
            is_customer,
            loc_facts,
            visited_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demand: demand_by_place,
            cost,
            via_cost,
            nearest_direct,
            nearest_via,
            avg_depot_cost,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_propo: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; n_places]),
        })
    }

    fn greedy_completion(
        &self,
        start: usize,
        load: f64,
        vehicles: f64,
        unvisited: &mut [bool],
        mut remaining_count: usize,
    ) -> f64 {
        let mut loc = start;
        let mut load_val = load;
        let mut veh = vehicles;
        let mut total = 0.0;

        while remaining_count > 0 {
            let mut best: Option<(f64, usize, bool)> = None;

            for &(cost, to) in &self.nearest_direct[loc] {
                if unvisited[to] && load_val + self.demand[to] <= self.capacity + 1e-9 {
                    best = Some((cost, to, false));
                    break;
                }
            }

            if self.is_customer[loc] && veh < self.max_vehicles {
                for &(cost, to) in &self.nearest_via[loc] {
                    if unvisited[to] {
                        if best.as_ref().map_or(true, |(best_cost, _, _)| cost < *best_cost) {
                            best = Some((cost, to, true));
                        }
                        break;
                    }
                }
            }

            match best {
                Some((cost, to, is_via)) => {
                    total += cost;
                    loc = to;
                    if is_via {
                        load_val = self.demand[to];
                        veh += 1.0;
                    } else {
                        load_val += self.demand[to];
                    }
                    unvisited[to] = false;
                    remaining_count -= 1;
                }
                None => return LARGE_PENALTY,
            }
        }

        if loc != self.depot_index {
            let ret = self.cost[loc][self.depot_index];
            if ret.is_finite() {
                total += ret;
            } else {
                return LARGE_PENALTY;
            }
        }

        total
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

        let mut numeric_values = self.scratch_numeric.borrow_mut();
        registry
            .view(state)
            .fill_numeric(&mut *numeric_values)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;
        let mut propositional_values = self.scratch_propo.borrow_mut();
        state.fill_state(registry, &mut *propositional_values);

        let load = numeric_values[self.load_id];
        let vehicles = numeric_values[self.vehicles_id];

        let mut current_loc = self.depot_index;
        let mut found_loc = false;
        for place_idx in 0..self.n_places {
            let fact = self.loc_facts[place_idx];
            if propositional_values[fact.var_id] == fact.value {
                current_loc = place_idx;
                found_loc = true;
                break;
            }
        }
        if !found_loc {
            return Ok(0.0);
        }

        {
            let mut unvisited = self.scratch_unvisited.borrow_mut();
            unvisited.fill(false);

            let mut unvisited_count = 0usize;
            let mut remaining_demand = 0.0;
            for &c in &self.customer_indices {
                let is_unvisited = match self.visited_facts[c] {
                    Some(fact) => propositional_values[fact.var_id] != fact.value,
                    None => true,
                };
                if is_unvisited {
                    unvisited[c] = true;
                    unvisited_count += 1;
                    remaining_demand += self.demand[c];
                }
            }

            if unvisited_count == 0 {
                let ret = self.cost[current_loc][self.depot_index];
                return Ok(if ret.is_finite() { ret } else { LARGE_PENALTY });
            }

            let residual = (self.capacity - load).max(0.0);
            let remaining_vehicles = self.max_vehicles - vehicles;
            if remaining_vehicles < -1e-9 {
                return Ok(LARGE_PENALTY);
            }
            let available = residual + remaining_vehicles.max(0.0) * self.capacity;
            if remaining_demand > available + 1e-9 {
                return Ok(LARGE_PENALTY);
            }

            let h_routing = self
                .greedy_completion(
                    current_loc,
                    load,
                    vehicles,
                    unvisited.as_mut_slice(),
                    unvisited_count,
                )
                .min(LARGE_PENALTY);

            let extra_routes = ((remaining_demand - residual).max(0.0)
                / self.capacity.max(1e-9))
                .ceil();
            let h_capacity = extra_routes * self.avg_depot_cost * 0.5;
            let slack = (remaining_vehicles - extra_routes).max(0.0);
            let h_vehicle = if slack < 1.0 { 5.0 * remaining_demand } else { 0.0 };

            Ok((h_routing + h_capacity + h_vehicle).min(LARGE_PENALTY))
        }
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
