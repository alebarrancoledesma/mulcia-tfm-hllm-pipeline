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
use std::collections::HashMap;

const LARGE_PENALTY: f64 = 1e12;
const INF_COST: f64 = 1e30;

struct LlmHeuristic {
    numeric_count: usize,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    n_customers: usize,
    depot_idx: usize,
    demands: Vec<f64>,
    loc_facts: Vec<ExplicitFact>,
    visited_facts: Vec<ExplicitFact>,
    direct_costs: Vec<Vec<f64>>,
    via_costs: Vec<Vec<f64>>,
    return_costs: Vec<f64>,
    nearest_from_place: Vec<Vec<usize>>,
    nearest_via_from_customer: Vec<Vec<usize>>,
    avg_direct_cost: f64,
    avg_via_cost: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

fn build_err(msg: &str) -> HeuristicBuildError {
    msg.to_string().into()
}

fn parse_pair_content(content: &str) -> Option<(&str, &str)> {
    if let Some(comma) = content.find(',') {
        let from = content[..comma].trim();
        let to = content[comma + 1..].trim();
        Some((from, to))
    } else {
        let mut parts = content.split_whitespace();
        let from = parts.next()?;
        let to = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        Some((from, to))
    }
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_var: Option<usize> = None;
        let mut visited_var: Option<usize> = None;
        let mut loc_values: HashMap<String, usize> = HashMap::new();
        let mut visited_values: HashMap<String, usize> = HashMap::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact);
                if let Some(rest) = name.strip_prefix("Atom loc(") {
                    if let Some(obj) = rest.strip_suffix(')') {
                        loc_var = Some(var_id);
                        loc_values.insert(obj.to_string(), value);
                    }
                } else if let Some(rest) = name.strip_prefix("Atom visited(") {
                    if let Some(obj) = rest.strip_suffix(')') {
                        visited_var = Some(var_id);
                        visited_values.insert(obj.to_string(), value);
                    }
                }
            }
        }

        let loc_var = loc_var.ok_or_else(|| build_err("Could not find loc proposition variable"))?;
        let visited_var = visited_var.ok_or_else(|| build_err("Could not find visited proposition variable"))?;

        let mut depot_name: Option<String> = None;
        for name in loc_values.keys() {
            if !visited_values.contains_key(name) {
                depot_name = Some(name.clone());
                break;
            }
        }
        let depot_name = depot_name.ok_or_else(|| build_err("Could not identify depot"))?;

        let mut customers: Vec<String> = visited_values.keys().cloned().collect();
        customers.sort();
        let n_customers = customers.len();
        if n_customers == 0 {
            return Err(build_err("No customers found"));
        }

        let mut customer_index: HashMap<String, usize> = HashMap::new();
        for (i, c) in customers.iter().enumerate() {
            customer_index.insert(c.clone(), i);
        }

        let mut place_index: HashMap<String, usize> = HashMap::new();
        place_index.insert(depot_name.clone(), 0);
        for (i, c) in customers.iter().enumerate() {
            place_index.insert(c.clone(), i + 1);
        }

        let places_len = n_customers + 1;
        let mut loc_facts = Vec::with_capacity(places_len);
        loc_facts.push(ExplicitFact::propositional(loc_var, loc_values[&depot_name]));
        for c in &customers {
            loc_facts.push(ExplicitFact::propositional(loc_var, loc_values[c]));
        }

        let visited_facts: Vec<ExplicitFact> = customers
            .iter()
            .map(|c| ExplicitFact::propositional(visited_var, visited_values[c]))
            .collect();

        let initial_numeric = task.get_initial_numeric_state_values();
        let numeric_count = task.numeric_variables().len();
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity: Option<f64> = None;
        let mut max_vehicles: Option<f64> = None;
        let mut demand_by_customer = vec![0.0; n_customers];

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let raw = var.name();
            let clean = raw.strip_prefix("PNE ").unwrap_or(raw).trim();
            if clean == "load()" {
                load_id = Some(id);
            } else if clean == "vehicles()" {
                vehicles_id = Some(id);
            } else if clean == "capacity()" {
                capacity = Some(initial_numeric.get(id).copied().unwrap_or(0.0));
            } else if clean == "max_vehicles()" || clean == "max-vehicles()" {
                max_vehicles = Some(initial_numeric.get(id).copied().unwrap_or(0.0));
            } else if let Some(pos) = clean.find("demand(") {
                let after = &clean[pos + 7..];
                if let Some(end) = after.find(')') {
                    let obj = after[..end].trim();
                    if let Some(&cust_idx) = customer_index.get(obj) {
                        demand_by_customer[cust_idx] = initial_numeric.get(id).copied().unwrap_or(0.0);
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| build_err("Could not find load() numeric variable"))?;
        let vehicles_id = vehicles_id.ok_or_else(|| build_err("Could not find vehicles() numeric variable"))?;
        let capacity = capacity.ok_or_else(|| build_err("Could not find capacity() numeric variable"))?;
        let max_vehicles = max_vehicles.ok_or_else(|| build_err("Could not find max_vehicles() numeric variable"))?;
        if capacity <= 0.0 || max_vehicles <= 0.0 {
            return Err(build_err("Non-positive capacity or max_vehicles"));
        }

        let mut direct_costs = vec![vec![INF_COST; n_customers]; places_len];
        let mut via_costs = vec![vec![INF_COST; n_customers]; n_customers];
        let mut return_costs = vec![INF_COST; n_customers];

        for op in task.get_operators() {
            let name = op.name();
            let parts: Vec<&str> = name.split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "visit" {
                if let (Some(&from_idx), Some(&to_idx)) = (place_index.get(parts[1]), customer_index.get(parts[2])) {
                    let cost = metric_operator_cost_from_initial_values(task, op);
                    if cost.is_finite() {
                        direct_costs[from_idx][to_idx] = cost;
                    }
                }
            } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
                if let (Some(&from_idx), Some(&to_idx)) = (customer_index.get(parts[1]), customer_index.get(parts[2])) {
                    let cost = metric_operator_cost_from_initial_values(task, op);
                    if cost.is_finite() {
                        via_costs[from_idx][to_idx] = cost;
                    }
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&cust_idx) = customer_index.get(parts[1]) {
                    let cost = metric_operator_cost_from_initial_values(task, op);
                    if cost.is_finite() {
                        return_costs[cust_idx] = cost;
                    }
                }
            }
        }

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let raw = var.name();
            let clean = raw.strip_prefix("PNE ").unwrap_or(raw).trim();
            let val = initial_numeric.get(id).copied().unwrap_or(0.0);
            if clean.contains("travel-cost-via-depot(") {
                if let Some(start) = clean.find("travel-cost-via-depot(") {
                    let after = &clean[start + "travel-cost-via-depot(".len()..];
                    if let Some(end) = after.find(')') {
                        let content = &after[..end];
                        if let Some((from_obj, to_obj)) = parse_pair_content(content) {
                            if let (Some(&fi), Some(&ti)) = (customer_index.get(from_obj), customer_index.get(to_obj)) {
                                if via_costs[fi][ti] >= INF_COST / 2.0 {
                                    via_costs[fi][ti] = val;
                                }
                            }
                        }
                    }
                }
            } else if clean.contains("travel-cost(") {
                if let Some(start) = clean.find("travel-cost(") {
                    let after = &clean[start + "travel-cost(".len()..];
                    if let Some(end) = after.find(')') {
                        let content = &after[..end];
                        if let Some((from_obj, to_obj)) = parse_pair_content(content) {
                            if let (Some(&fi), Some(&ti)) = (place_index.get(from_obj), customer_index.get(to_obj)) {
                                if direct_costs[fi][ti] >= INF_COST / 2.0 {
                                    direct_costs[fi][ti] = val;
                                }
                            } else if let (Some(&fi), Some(&ti)) = (customer_index.get(from_obj), customer_index.get(to_obj)) {
                                if direct_costs[fi + 1][ti] >= INF_COST / 2.0 {
                                    direct_costs[fi + 1][ti] = val;
                                }
                            }
                        }
                    }
                }
            }
        }

        for c in 0..n_customers {
            if direct_costs[0][c] >= INF_COST / 2.0 {
                return Err(build_err("Missing depot->customer travel cost"));
            }
            if return_costs[c] >= INF_COST / 2.0 {
                return Err(build_err("Missing customer->depot return cost"));
            }
        }

        for i in 0..n_customers {
            for j in 0..n_customers {
                if via_costs[i][j] >= INF_COST / 2.0 {
                    via_costs[i][j] = return_costs[i] + direct_costs[0][j];
                }
                if direct_costs[i + 1][j] >= INF_COST / 2.0 {
                    direct_costs[i + 1][j] = return_costs[i] + direct_costs[0][j];
                }
            }
        }

        let mut nearest_from_place: Vec<Vec<usize>> = Vec::with_capacity(places_len);
        for p in 0..places_len {
            let mut list: Vec<usize> = (0..n_customers).collect();
            list.sort_by(|&a, &b| {
                direct_costs[p][a]
                    .partial_cmp(&direct_costs[p][b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_from_place.push(list);
        }

        let mut nearest_via_from_customer: Vec<Vec<usize>> = Vec::with_capacity(n_customers);
        for i in 0..n_customers {
            let mut list: Vec<usize> = (0..n_customers).collect();
            list.sort_by(|&a, &b| {
                via_costs[i][a]
                    .partial_cmp(&via_costs[i][b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_via_from_customer.push(list);
        }

        let mut sum_direct = 0.0;
        let mut cnt_direct = 0;
        for p in 0..places_len {
            for c in 0..n_customers {
                if direct_costs[p][c] < INF_COST / 2.0 {
                    sum_direct += direct_costs[p][c];
                    cnt_direct += 1;
                }
            }
        }

        let mut sum_via = 0.0;
        let mut cnt_via = 0;
        for i in 0..n_customers {
            for j in 0..n_customers {
                if via_costs[i][j] < INF_COST / 2.0 {
                    sum_via += via_costs[i][j];
                    cnt_via += 1;
                }
            }
        }

        let avg_direct_cost = if cnt_direct > 0 { sum_direct / cnt_direct as f64 } else { 1.0 };
        let avg_via_cost = if cnt_via > 0 { sum_via / cnt_via as f64 } else { 1.0 };

        Ok(Self {
            numeric_count,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            n_customers,
            depot_idx: 0,
            demands: demand_by_customer,
            loc_facts,
            visited_facts,
            direct_costs,
            via_costs,
            return_costs,
            nearest_from_place,
            nearest_via_from_customer,
            avg_direct_cost,
            avg_via_cost,
            scratch_numeric: RefCell::new(vec![0.0; numeric_count]),
            scratch_unvisited: RefCell::new(vec![false; n_customers]),
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

        {
            let mut numeric_values = self.scratch_numeric.borrow_mut();
            numeric_values.resize(self.numeric_count, 0.0);
            registry.view(state).fill_numeric(&mut numeric_values);
            let load = numeric_values[self.load_id];
            let vehicles = numeric_values[self.vehicles_id];

            let mut current_place = self.depot_idx;
            for (idx, fact) in self.loc_facts.iter().enumerate() {
                if fact.is_hold(registry.view(state)) {
                    current_place = idx;
                    break;
                }
            }

            let mut unvisited = self.scratch_unvisited.borrow_mut();
            unvisited.clear();
            unvisited.resize(self.n_customers, true);
            let mut remaining_customers = 0usize;
            for (idx, fact) in self.visited_facts.iter().enumerate() {
                if fact.is_hold(registry.view(state)) {
                    unvisited[idx] = false;
                } else {
                    remaining_customers += 1;
                }
            }

            if remaining_customers == 0 {
                if current_place != self.depot_idx {
                    let ret = self.return_costs[current_place - 1];
                    return Ok(if ret < INF_COST / 2.0 { ret } else { LARGE_PENALTY });
                } else {
                    return Ok(0.0);
                }
            }

            if load > self.capacity + 1e-9 || vehicles > self.max_vehicles + 1e-9 {
                return Ok(LARGE_PENALTY);
            }

            let remaining_demand: f64 = unvisited
                .iter()
                .enumerate()
                .filter_map(|(idx, unv)| if *unv { Some(self.demands[idx]) } else { None })
                .sum();

            let residual_capacity = (self.capacity - load).max(0.0);
            let extra_capacity = (self.max_vehicles - vehicles).max(0.0) * self.capacity;
            let available_capacity = residual_capacity + extra_capacity;

            if remaining_demand > available_capacity + 1e-9 {
                return Ok(LARGE_PENALTY);
            }

            let mut cur = current_place;
            let mut cur_load = load;
            let mut used_vehicles = vehicles;
            let mut route_cost = 0.0;
            let mut feasible = true;

            while remaining_customers > 0 {
                let residual = (self.capacity - cur_load).max(0.0);

                let direct_candidate = self.nearest_from_place[cur]
                    .iter()
                    .copied()
                    .find(|&c| {
                        unvisited[c]
                            && self.demands[c] <= residual + 1e-9
                            && self.direct_costs[cur][c] < INF_COST / 2.0
                    });

                let via_candidate = if cur != self.depot_idx && used_vehicles < self.max_vehicles {
                    self.nearest_via_from_customer[cur - 1]
                        .iter()
                        .copied()
                        .find(|&c| {
                            unvisited[c]
                                && self.demands[c] <= self.capacity + 1e-9
                                && self.via_costs[cur - 1][c] < INF_COST / 2.0
                        })
                } else {
                    None
                };

                let (pick, via_flag, added_cost) = match (direct_candidate, via_candidate) {
                    (Some(d), Some(v)) => {
                        let dc = self.direct_costs[cur][d];
                        let vc = self.via_costs[cur - 1][v];
                        if vc < dc {
                            (v, true, vc)
                        } else {
                            (d, false, dc)
                        }
                    }
                    (Some(d), None) => (d, false, self.direct_costs[cur][d]),
                    (None, Some(v)) => (v, true, self.via_costs[cur - 1][v]),
                    (None, None) => {
                        feasible = false;
                        break;
                    }
                };

                route_cost += added_cost;
                unvisited[pick] = false;
                remaining_customers -= 1;

                if via_flag {
                    used_vehicles += 1.0;
                    cur_load = self.demands[pick];
                } else {
                    cur_load += self.demands[pick];
                }

                cur = pick + 1;
            }

            if !feasible {
                return Ok(LARGE_PENALTY + route_cost);
            }

            if cur != self.depot_idx {
                let ret = self.return_costs[cur - 1];
                route_cost += if ret < INF_COST / 2.0 { ret } else { 0.0 };
            }

            let required_new_vehicles = ((remaining_demand - residual_capacity)
                / self.capacity.max(1.0))
                .ceil()
                .max(0.0);

            let h_fleet = self.avg_via_cost * required_new_vehicles * 0.35;
            let h_capacity = (remaining_demand / self.capacity.max(1.0))
                * self.avg_direct_cost
                * 0.15;

            Ok(route_cost + h_fleet + h_capacity)
        }
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
