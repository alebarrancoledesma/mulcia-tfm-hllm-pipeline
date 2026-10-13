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
use std::collections::HashMap;

// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

struct LlmHeuristic {
    loc_facts: Vec<(String, ExplicitFact)>,
    visited_facts: Vec<ExplicitFact>,
    customer_names: Vec<String>,
    name_to_idx: HashMap<String, usize>,
    depot_idx: usize,
    dist: Vec<Vec<f64>>,
    via_cost: Vec<Vec<f64>>,
    direct_neighbors: Vec<Vec<usize>>,
    via_neighbors: Vec<Vec<usize>>,
    demands: Vec<f64>,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    large_penalty: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_remaining: RefCell<Vec<bool>>,
}

fn extract_arg(name: &str, predicate: &str) -> Option<String> {
    let idx = name.find(predicate)?;
    let start = idx + predicate.len();
    let end = start + name[start..].find(')')?;
    Some(name[start..end].to_string())
}

fn parse_visit_name(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix("visit ")?;
    let mut parts = rest.split_whitespace();
    let from = parts.next()?.to_string();
    let to = parts.next()?.to_string();
    Some((from, to))
}

fn parse_visit_via_depot_name(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix("visit-via-depot ")?;
    let mut parts = rest.split_whitespace();
    let from = parts.next()?.to_string();
    let to = parts.next()?.to_string();
    Some((from, to))
}

fn parse_return_to_depot_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("return-to-depot ")?;
    rest.split_whitespace().next().map(|s| s.to_string())
}

fn parse_travel_cost_arg(name: &str) -> Option<(String, String)> {
    let idx = name.find("travel-cost(")?;
    let start = idx + "travel-cost(".len();
    let end = start + name[start..].find(')')?;
    let args_str = &name[start..end];

    let mut comma_parts = args_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty());
    if let (Some(a), Some(b)) = (comma_parts.next(), comma_parts.next()) {
        return Some((a.to_string(), b.to_string()));
    }

    let mut ws_parts = args_str.split_whitespace();
    match (ws_parts.next(), ws_parts.next()) {
        (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
        _ => None,
    }
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_pairs: Vec<(String, ExplicitFact)> = Vec::new();
        let mut visited_pairs: Vec<(String, ExplicitFact)> = Vec::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(name) = extract_arg(&fact_name, "loc(") {
                    loc_pairs.push((name, fact));
                } else if let Some(name) = extract_arg(&fact_name, "visited(") {
                    visited_pairs.push((name, fact));
                }
            }
        }

        if loc_pairs.is_empty() {
            return Err("No loc facts found".to_string().into());
        }

        let mut visited_names: HashMap<String, bool> = HashMap::new();
        for (name, _) in &visited_pairs {
            visited_names.insert(name.clone(), true);
        }

        let depot_name = loc_pairs
            .iter()
            .find(|(name, _)| !visited_names.contains_key(name))
            .map(|(name, _)| name.clone())
            .ok_or_else(|| "Cannot determine depot structurally".to_string().into())?;

        let mut name_to_idx = HashMap::new();
        let mut place_names = Vec::new();
        name_to_idx.insert(depot_name.clone(), 0);
        place_names.push(depot_name.clone());
        let depot_idx = 0;

        let mut customer_names = Vec::new();
        let mut visited_facts = Vec::new();
        for (name, fact) in &visited_pairs {
            let idx = place_names.len();
            name_to_idx.insert(name.clone(), idx);
            place_names.push(name.clone());
            customer_names.push(name.clone());
            visited_facts.push(fact.clone());
        }

        let n_places = place_names.len();
        let mut dist = vec![vec![f64::INFINITY; n_places]; n_places];
        let mut via_cost = vec![vec![f64::INFINITY; n_places]; n_places];
        for i in 0..n_places {
            dist[i][i] = 0.0;
            via_cost[i][i] = 0.0;
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = f64::INFINITY;
        let mut max_vehicles = f64::INFINITY;
        let mut demands = vec![0.0; customer_names.len()];

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().trim().to_string();
            if name == "load()" || name == "PNE load()" {
                load_id = Some(id);
            } else if name == "vehicles()" || name == "PNE vehicles()" {
                vehicles_id = Some(id);
            } else if name == "capacity()" || name == "PNE capacity()" {
                capacity = initial_numeric[id];
            } else if name == "max_vehicles()" || name == "PNE max_vehicles()" {
                max_vehicles = initial_numeric[id];
            } else {
                for (ci, cust) in customer_names.iter().enumerate() {
                    if name == format!("demand({})", cust)
                        || name == format!("PNE demand({})", cust)
                    {
                        demands[ci] = initial_numeric[id];
                        break;
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| "Missing load() variable".to_string().into())?;
        let vehicles_id =
            vehicles_id.ok_or_else(|| "Missing vehicles() variable".to_string().into())?;

        // Phase A: extract costs from grounded operators.
        for op in task.get_operators() {
            let op_name = op.name().to_string();
            let cost = metric_operator_cost_from_initial_values(task, op);

            if let Some((from, to)) = parse_visit_via_depot_name(&op_name) {
                if let (Some(&fi), Some(&ti)) = (name_to_idx.get(&from), name_to_idx.get(&to)) {
                    via_cost[fi][ti] = via_cost[fi][ti].min(cost);
                }
            } else if let Some((from, to)) = parse_visit_name(&op_name) {
                if let (Some(&fi), Some(&ti)) = (name_to_idx.get(&from), name_to_idx.get(&to)) {
                    dist[fi][ti] = dist[fi][ti].min(cost);
                }
            } else if let Some(from) = parse_return_to_depot_name(&op_name) {
                if let Some(&fi) = name_to_idx.get(&from) {
                    dist[fi][depot_idx] = dist[fi][depot_idx].min(cost);
                }
            }
        }

        // Phase B (optional fallback): recover still-missing travel-cost values.
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if name.contains("travel-cost(") {
                if let Some((from, to)) = parse_travel_cost_arg(&name) {
                    if let (Some(&fi), Some(&ti)) = (name_to_idx.get(&from), name_to_idx.get(&to)) {
                        if !dist[fi][ti].is_finite() {
                            dist[fi][ti] = initial_numeric[id];
                        }
                    }
                }
            }
        }

        // Phase C: fill missing arcs through the depot when both legs are known.
        for i in 0..n_places {
            for j in 0..n_places {
                if i == j {
                    continue;
                }
                if !dist[i][j].is_finite()
                    && dist[i][depot_idx].is_finite()
                    && dist[depot_idx][j].is_finite()
                {
                    dist[i][j] = dist[i][depot_idx] + dist[depot_idx][j];
                }
                if !via_cost[i][j].is_finite()
                    && dist[i][depot_idx].is_finite()
                    && dist[depot_idx][j].is_finite()
                {
                    via_cost[i][j] = dist[i][depot_idx] + dist[depot_idx][j];
                }
            }
        }

        for (ci, cust) in customer_names.iter().enumerate() {
            let cust_idx = ci + 1;
            if !dist[depot_idx][cust_idx].is_finite() {
                return Err(format!(
                    "Missing travel cost from depot to customer {}",
                    cust
                )
                .into());
            }
            if !dist[cust_idx][depot_idx].is_finite() {
                return Err(format!(
                    "Missing travel cost from customer {} to depot",
                    cust
                )
                .into());
            }
        }

        let mut direct_neighbors = vec![Vec::new(); n_places];
        let mut via_neighbors = vec![Vec::new(); n_places];
        for i in 0..n_places {
            for j in 0..n_places {
                if i != j && dist[i][j].is_finite() {
                    direct_neighbors[i].push(j);
                }
                if i != j && via_cost[i][j].is_finite() {
                    via_neighbors[i].push(j);
                }
            }
            direct_neighbors[i].sort_by(|&a, &b| {
                dist[i][a]
                    .partial_cmp(&dist[i][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            via_neighbors[i].sort_by(|&a, &b| {
                via_cost[i][a]
                    .partial_cmp(&via_cost[i][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(Self {
            loc_facts: loc_pairs,
            visited_facts,
            customer_names,
            name_to_idx,
            depot_idx,
            dist,
            via_cost,
            direct_neighbors,
            via_neighbors,
            demands,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            large_penalty: 1.0e12,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_remaining: RefCell::new(vec![false; n_places]),
        })
    }

    fn greedy_completion(
        &self,
        start_idx: usize,
        start_load: f64,
        start_vehicles: f64,
        initial_unvisited: &[usize],
    ) -> f64 {
        let mut remaining = self.scratch_remaining.borrow_mut();
        for v in remaining.iter_mut() {
            *v = false;
        }
        for &idx in initial_unvisited {
            remaining[idx] = true;
        }

        let mut cur = start_idx;
        let mut load = start_load;
        let mut veh = start_vehicles;
        let mut total = 0.0;
        let mut remaining_count = initial_unvisited.len();

        while remaining_count > 0 {
            let mut best_direct: Option<(usize, f64)> = None;
            for &to_idx in &self.direct_neighbors[cur] {
                if remaining[to_idx] && load + self.demands[to_idx - 1] <= self.capacity {
                    best_direct = Some((to_idx, self.dist[cur][to_idx]));
                    break;
                }
            }

            let mut best_via: Option<(usize, f64)> = None;
            if cur != self.depot_idx && veh + 1.0 <= self.max_vehicles {
                for &to_idx in &self.via_neighbors[cur] {
                    if remaining[to_idx] && self.demands[to_idx - 1] <= self.capacity {
                        best_via = Some((to_idx, self.via_cost[cur][to_idx]));
                        break;
                    }
                }
            }

            if let Some((to_idx, cost)) = best_direct {
                total += cost;
                load += self.demands[to_idx - 1];
                cur = to_idx;
                remaining[to_idx] = false;
                remaining_count -= 1;
            } else if let Some((to_idx, cost)) = best_via {
                total += cost;
                load = self.demands[to_idx - 1];
                veh += 1.0;
                cur = to_idx;
                remaining[to_idx] = false;
                remaining_count -= 1;
            } else {
                return self.large_penalty;
            }
        }

        if cur != self.depot_idx {
            let cost = self.dist[cur][self.depot_idx];
            if !cost.is_finite() {
                return self.large_penalty;
            }
            total += cost;
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

        let (load, vehicles) = {
            let mut nums = self.scratch_numeric.borrow_mut();
            registry.view(state).fill_numeric(&mut nums).map_err(|err| {
                EvaluationError::ComputationFailed(format!("{err:?}"))
            })?;
            (nums[self.load_id], nums[self.vehicles_id])
        };

        let mut cur_idx = None;
        for (name, fact) in &self.loc_facts {
            if fact.is_hold(registry.view(state)) {
                if let Some(&idx) = self.name_to_idx.get(name) {
                    cur_idx = Some(idx);
                    break;
                }
            }
        }

        let cur_idx = match cur_idx {
            Some(idx) => idx,
            None => return Ok(self.large_penalty),
        };

        let mut unvisited = Vec::new();
        for (i, fact) in self.visited_facts.iter().enumerate() {
            if !fact.is_hold(registry.view(state)) {
                unvisited.push(i + 1);
            }
        }

        if unvisited.is_empty() {
            if cur_idx == self.depot_idx {
                return Ok(0.0);
            }
            let cost = self.dist[cur_idx][self.depot_idx];
            return Ok(if cost.is_finite() {
                cost
            } else {
                self.large_penalty
            });
        }

        let remaining_demand: f64 = unvisited.iter().map(|&idx| self.demands[idx - 1]).sum();

        let current_remaining = if self.capacity.is_finite() {
            (self.capacity - load).max(0.0)
        } else {
            f64::INFINITY
        };

        let remaining_fleet_cap =
            if self.max_vehicles.is_finite() && self.capacity.is_finite() {
                (self.max_vehicles - vehicles).max(0.0) * self.capacity
            } else if self.max_vehicles.is_finite() {
                f64::INFINITY
            } else {
                f64::INFINITY
            };

        let total_available = current_remaining + remaining_fleet_cap;

        if remaining_demand > total_available {
            return Ok(self.large_penalty + (remaining_demand - total_available));
        }

        if self.max_vehicles.is_finite() && vehicles > self.max_vehicles {
            return Ok(self.large_penalty + 1.0);
        }

        let h_routing = self.greedy_completion(cur_idx, load, vehicles, &unvisited);

        Ok(h_routing)
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
