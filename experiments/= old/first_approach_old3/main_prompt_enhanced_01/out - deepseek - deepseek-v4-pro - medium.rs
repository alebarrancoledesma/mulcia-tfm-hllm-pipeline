use std::cell::RefCell;
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

fn extract_paren(text: &str, prefix: &str) -> Option<String> {
    let token = format!("{}(", prefix);
    let start = text.find(&token)? + token.len();
    let end = text[start..].find(')')? + start;
    Some(text[start..end].trim().to_string())
}

fn parse_two_place_names(inner: &str) -> Option<(String, String)> {
    let cleaned = inner.replace(',', ' ');
    let parts: Vec<&str> = cleaned.split_whitespace().collect();
    if parts.len() == 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

fn place_index(
    name: &str,
    customer_idx: &HashMap<String, usize>,
    depot_name: &str,
    depot_idx: usize,
) -> Option<usize> {
    if name == depot_name {
        Some(depot_idx)
    } else {
        customer_idx.get(name).copied()
    }
}

struct LlmHeuristic {
    n: usize,
    depot_idx: usize,
    visited_facts: Vec<Option<ExplicitFact>>,
    loc_facts: Vec<(usize, ExplicitFact)>,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,
    direct: Vec<Vec<f64>>,
    via_depot: Vec<Vec<f64>>,
    return_cost: Vec<f64>,
    nearest_customers_from: Vec<Vec<(f64, usize)>>,
    via_customers_from: Vec<Vec<(f64, usize)>>,
    avg_depot_arc: f64,
    large_penalty: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_visited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let variables = task.variables();
        let mut loc_raw: Vec<(ExplicitFact, String)> = Vec::new();
        let mut visited_raw: Vec<(ExplicitFact, String)> = Vec::new();
        let mut visited_name_set: BTreeSet<String> = BTreeSet::new();

        for (var_id, var) in variables.iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact).to_string();
                if let Some(p) = extract_paren(&name, "loc") {
                    loc_raw.push((fact.clone(), p));
                } else if let Some(c) = extract_paren(&name, "visited") {
                    visited_raw.push((fact.clone(), c.clone()));
                    visited_name_set.insert(c);
                }
            }
        }

        let customer_names: Vec<String> = visited_name_set.into_iter().collect();
        let n = customer_names.len();
        let mut customer_idx: HashMap<String, usize> = HashMap::new();
        for (i, c) in customer_names.iter().enumerate() {
            customer_idx.insert(c.clone(), i);
        }

        let depot_name = loc_raw
            .iter()
            .map(|(_, p)| p.clone())
            .find(|p| !customer_idx.contains_key(p))
            .ok_or_else(|| {
                "unable to identify depot from loc facts".to_string().into()
            })?;
        let depot_idx = n;

        let mut visited_facts: Vec<Option<ExplicitFact>> = vec![None; n];
        for (fact, cname) in &visited_raw {
            if let Some(&idx) = customer_idx.get(cname) {
                visited_facts[idx] = Some(fact.clone());
            }
        }

        let mut loc_facts: Vec<(usize, ExplicitFact)> = Vec::new();
        for (fact, pname) in &loc_raw {
            let idx = if let Some(&idx) = customer_idx.get(pname) {
                idx
            } else {
                depot_idx
            };
            loc_facts.push((idx, fact.clone()));
        }

        let numeric_variables = task.numeric_variables();
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity_val: Option<f64> = None;
        let mut max_vehicles_val: Option<f64> = None;
        let mut demand_vals: Vec<Option<f64>> = vec![None; n];

        for (id, var) in numeric_variables.iter().enumerate() {
            let name = var.name().to_string();
            let val = initial_numeric.get(id).copied().unwrap_or(0.0);

            if load_id.is_none() && name.contains("load()") {
                load_id = Some(id);
            }
            if vehicles_id.is_none() && name.contains("vehicles()") {
                vehicles_id = Some(id);
            }
            if name.contains("capacity") && !name.contains("max_vehicles") && capacity_val.is_none() {
                capacity_val = Some(val);
            }
            if name.contains("max_vehicles") && max_vehicles_val.is_none() {
                max_vehicles_val = Some(val);
            }
            if let Some(cname) = extract_paren(&name, "demand") {
                if let Some(&idx) = customer_idx.get(&cname) {
                    if demand_vals[idx].is_none() {
                        demand_vals[idx] = Some(val);
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            "missing load() numeric variable".to_string().into()
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            "missing vehicles() numeric variable".to_string().into()
        })?;

        let capacity = capacity_val.unwrap_or(1e9);
        let max_vehicles = max_vehicles_val.unwrap_or(1e9);
        let demands: Vec<f64> = demand_vals.iter().map(|v| v.unwrap_or(0.0)).collect();

        let mut direct = vec![vec![f64::INFINITY; n + 1]; n + 1];
        let mut via_depot = vec![vec![f64::INFINITY; n]; n];
        let mut return_cost = vec![f64::INFINITY; n];

        let operators = task.get_operators();
        for op in &operators {
            let name = op.name().to_string();
            let cost = metric_operator_cost_from_initial_values(task, op);
            let parts: Vec<&str> = name.split_whitespace().collect();

            if parts.len() == 3 && parts[0] == "visit" {
                if let (Some(f), Some(t)) = (
                    place_index(parts[1], &customer_idx, &depot_name, depot_idx),
                    place_index(parts[2], &customer_idx, &depot_name, depot_idx),
                ) {
                    direct[f][t] = if direct[f][t].is_finite() {
                        direct[f][t].min(cost)
                    } else {
                        cost
                    };
                }
            } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
                if let (Some(f), Some(t)) = (
                    customer_idx.get(parts[1]).copied(),
                    customer_idx.get(parts[2]).copied(),
                ) {
                    via_depot[f][t] = if via_depot[f][t].is_finite() {
                        via_depot[f][t].min(cost)
                    } else {
                        cost
                    };
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&f) = customer_idx.get(parts[1]) {
                    return_cost[f] = if return_cost[f].is_finite() {
                        return_cost[f].min(cost)
                    } else {
                        cost
                    };
                }
            }
        }

        for (id, var) in numeric_variables.iter().enumerate() {
            let name = var.name().to_string();
            let val = initial_numeric.get(id).copied().unwrap_or(0.0);

            if name.contains("travel-cost-via-depot(") {
                if let Some(inner) = extract_paren(&name, "travel-cost-via-depot") {
                    if let Some((a, b)) = parse_two_place_names(&inner) {
                        if let (Some(f), Some(t)) = (
                            customer_idx.get(&a).copied(),
                            customer_idx.get(&b).copied(),
                        ) {
                            if via_depot[f][t].is_infinite() {
                                via_depot[f][t] = val;
                            }
                        }
                    }
                }
            } else if name.contains("travel-cost(") {
                if let Some(inner) = extract_paren(&name, "travel-cost") {
                    if let Some((a, b)) = parse_two_place_names(&inner) {
                        let f = place_index(&a, &customer_idx, &depot_name, depot_idx);
                        let t = place_index(&b, &customer_idx, &depot_name, depot_idx);
                        if let (Some(f), Some(t)) = (f, t) {
                            if direct[f][t].is_infinite() {
                                direct[f][t] = val;
                            }
                            if f < n && t == depot_idx {
                                return_cost[f] = return_cost[f].min(val);
                            }
                        }
                    }
                }
            }
        }

        for i in 0..n {
            if return_cost[i].is_infinite() && direct[i][depot_idx].is_finite() {
                return_cost[i] = direct[i][depot_idx];
            }
        }

        for i in 0..n {
            if return_cost[i].is_infinite() || direct[depot_idx][i].is_infinite() {
                continue;
            }
            for j in 0..n {
                if i != j && via_depot[i][j].is_infinite() {
                    via_depot[i][j] = return_cost[i] + direct[depot_idx][j];
                }
            }
        }

        let has_depot_out = (0..n).any(|i| direct[depot_idx][i].is_finite());
        if !has_depot_out && (0..n).all(|i| via_depot[i].iter().all(|c| c.is_infinite())) {
            return Err(
                "no usable travel costs after operator and numeric extraction".to_string().into(),
            );
        }

        let mut nearest_customers_from: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n + 1);
        for from in 0..=n {
            let mut list: Vec<(f64, usize)> = Vec::new();
            for to in 0..n {
                if direct[from][to].is_finite() {
                    list.push((direct[from][to], to));
                }
            }
            list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            nearest_customers_from.push(list);
        }

        let mut via_customers_from: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n);
        for from in 0..n {
            let mut list: Vec<(f64, usize)> = Vec::new();
            for to in 0..n {
                if from == to {
                    continue;
                }
                let cost = if via_depot[from][to].is_finite() {
                    via_depot[from][to]
                } else {
                    let a = return_cost[from];
                    let b = direct[depot_idx][to];
                    if a.is_finite() && b.is_finite() {
                        a + b
                    } else {
                        f64::INFINITY
                    }
                };
                if cost.is_finite() {
                    list.push((cost, to));
                }
            }
            list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            via_customers_from.push(list);
        }

        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            if direct[depot_idx][i].is_finite() {
                sum += direct[depot_idx][i];
                count += 1;
            }
            if return_cost[i].is_finite() {
                sum += return_cost[i];
                count += 1;
            }
        }
        let avg_depot_arc = if count > 0 { sum / count as f64 } else { 1.0 };

        Ok(LlmHeuristic {
            n,
            depot_idx,
            visited_facts,
            loc_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            direct,
            via_depot,
            return_cost,
            nearest_customers_from,
            via_customers_from,
            avg_depot_arc,
            large_penalty: 1e12,
            scratch_numeric: RefCell::new(Vec::new()),
            scratch_visited: RefCell::new(Vec::new()),
        })
    }

    fn final_return_cost(&self, from_idx: usize) -> f64 {
        if from_idx == self.depot_idx {
            return 0.0;
        }
        if from_idx < self.n {
            if self.return_cost[from_idx].is_finite() {
                return self.return_cost[from_idx];
            }
            if self.direct[from_idx][self.depot_idx].is_finite() {
                return self.direct[from_idx][self.depot_idx];
            }
            return self.avg_depot_arc;
        }
        0.0
    }

    fn greedy_cost(
        &self,
        start_idx: usize,
        start_load: f64,
        start_vehicles: f64,
        visited: &mut [bool],
    ) -> Result<f64, ()> {
        let mut current = start_idx;
        let mut load = start_load;
        let mut vehicles = start_vehicles;
        let mut cost = 0.0;
        let mut remaining = visited.iter().filter(|&&v| !v).count();

        while remaining > 0 {
            if current == self.depot_idx {
                let mut chosen: Option<(f64, usize)> = None;
                for &(c, to) in &self.nearest_customers_from[current] {
                    if !visited[to] && load + self.demands[to] <= self.capacity + 1e-9 {
                        chosen = Some((c, to));
                        break;
                    }
                }
                if let Some((c, to)) = chosen {
                    cost += c;
                    current = to;
                    load += self.demands[to];
                    visited[to] = true;
                    remaining -= 1;
                } else {
                    return Err(());
                }
            } else {
                let mut chosen_direct: Option<(f64, usize)> = None;
                for &(c, to) in &self.nearest_customers_from[current] {
                    if !visited[to] && load + self.demands[to] <= self.capacity + 1e-9 {
                        chosen_direct = Some((c, to));
                        break;
                    }
                }

                if let Some((c, to)) = chosen_direct {
                    cost += c;
                    current = to;
                    load += self.demands[to];
                    visited[to] = true;
                    remaining -= 1;
                } else {
                    if vehicles < self.max_vehicles {
                        let mut chosen_via: Option<(f64, usize)> = None;
                        for &(c, to) in &self.via_customers_from[current] {
                            if !visited[to] && self.demands[to] <= self.capacity + 1e-9 {
                                chosen_via = Some((c, to));
                                break;
                            }
                        }
                        if let Some((c, to)) = chosen_via {
                            cost += c;
                            current = to;
                            load = self.demands[to];
                            visited[to] = true;
                            vehicles += 1.0;
                            remaining -= 1;
                        } else {
                            return Err(());
                        }
                    } else {
                        return Err(());
                    }
                }
            }
        }

        if current != self.depot_idx {
            cost += self.final_return_cost(current);
        }
        Ok(cost)
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
        numeric_values.resize(task.numeric_variables().len(), 0.0);
        registry
            .view(state)
            .fill_numeric(&mut numeric_values)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

        let load = numeric_values[self.load_id];
        let vehicles = numeric_values[self.vehicles_id];

        let mut current_idx = self.depot_idx;
        for (idx, fact) in &self.loc_facts {
            if fact.is_hold(registry.view(state)) {
                current_idx = *idx;
                break;
            }
        }

        let mut visited = self.scratch_visited.borrow_mut();
        visited.clear();
        visited.resize(self.n, false);

        let mut remaining_demand = 0.0;
        let mut unvisited_count = 0usize;

        for i in 0..self.n {
            let is_visited = self.visited_facts[i]
                .as_ref()
                .map(|f| f.is_hold(registry.view(state)))
                .unwrap_or(false);
            visited[i] = is_visited;
            if !is_visited {
                remaining_demand += self.demands[i];
                unvisited_count += 1;
            }
        }

        if unvisited_count == 0 {
            if current_idx == self.depot_idx {
                return Ok(0.0);
            }
            return Ok(self.final_return_cost(current_idx));
        }

        let current_remaining_capacity = (self.capacity - load).max(0.0);
        let remaining_vehicles = (self.max_vehicles - vehicles).max(0.0);
        let fleet_remaining_capacity =
            current_remaining_capacity + remaining_vehicles * self.capacity;

        if remaining_demand > fleet_remaining_capacity + 1e-9 {
            return Ok(self.large_penalty);
        }

        let extra_vehicles_needed = if remaining_demand > current_remaining_capacity {
            ((remaining_demand - current_remaining_capacity) / self.capacity).ceil()
        } else {
            0.0
        };

        let h_resource = extra_vehicles_needed * 2.0 * self.avg_depot_arc;

        let h_route = match self.greedy_cost(current_idx, load, vehicles, &mut visited) {
            Ok(cost) => cost,
            Err(()) => {
                let mut fallback = 0.0;
                if current_idx != self.depot_idx {
                    fallback += self.final_return_cost(current_idx);
                }
                fallback += unvisited_count as f64 * 2.0 * self.avg_depot_arc;
                fallback
            }
        };

        let h = h_route + h_resource;
        if h.is_finite() {
            Ok(h.max(0.0))
        } else {
            Ok(self.large_penalty)
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
    planforge::run_with_heuristics(vec![ExternalHeuristic {
        name: "llm_001",
        backend: RequiredBackend::None,
        requirements: any_task,
        nested_heuristics: no_nested_heuristics,
        build: build_llm_001,
    }])
}
