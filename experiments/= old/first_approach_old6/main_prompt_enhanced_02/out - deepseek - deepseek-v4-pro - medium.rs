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

const INF: f64 = 1e30;
const LARGE_PENALTY: f64 = 1e12;

#[derive(Clone, Copy)]
struct LocFact { place: usize, var_id: usize, value: usize }

#[derive(Clone, Copy)]
struct VisitedFact { place: usize, var_id: usize, value: usize }

#[derive(Clone, Copy, PartialEq)]
enum NextAction { Direct, ViaDepot }

fn parse_predicate_arg(name: &str, predicate: &str) -> Option<String> {
    let marker = predicate.to_owned() + "(";
    let start = name.find(marker.as_str())?;
    let rest = &name[start + marker.len()..];
    let end = rest.find(')')?;
    Some(rest[..end].to_string())
}

fn parse_travel_cost_name(name: &str) -> Option<(String, String)> {
    let marker = "travel-cost(";
    let start = name.find(marker)?;
    let rest = &name[start + marker.len()..];
    let end = rest.find(')')?;
    let inner = &rest[..end];
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
    depot: usize,
    customers: Vec<usize>,
    loc_facts: Vec<LocFact>,
    visited_facts: Vec<VisitedFact>,
    numeric_var_count: usize,
    prop_var_count: usize,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,
    direct_costs: Vec<Vec<f64>>,
    via_depot_costs: Vec<Vec<f64>>,
    nearest_direct: Vec<Vec<(usize, f64)>>,
    nearest_via: Vec<Vec<(usize, f64)>>,
    avg_depot_round_trip: f64,
    vehicle_use_penalty: f64,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_prop: RefCell<Vec<usize>>,
    scratch_visited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        fn intern(
            name: &str,
            map: &mut HashMap<String, usize>,
            names: &mut Vec<String>,
        ) -> usize {
            if let Some(&id) = map.get(name) {
                return id;
            }
            let id = names.len();
            names.push(name.to_string());
            map.insert(name.to_string(), id);
            id
        }

        let mut name_to_id: HashMap<String, usize> = HashMap::new();
        let mut names: Vec<String> = Vec::new();
        let mut loc_facts: Vec<LocFact> = Vec::new();
        let mut visited_facts: Vec<VisitedFact> = Vec::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size() as usize;
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(arg) = parse_predicate_arg(&fact_name, "loc") {
                    let p = intern(&arg, &mut name_to_id, &mut names);
                    loc_facts.push(LocFact { place: p, var_id, value });
                } else if let Some(arg) = parse_predicate_arg(&fact_name, "visited") {
                    let p = intern(&arg, &mut name_to_id, &mut names);
                    visited_facts.push(VisitedFact { place: p, var_id, value });
                }
            }
        }

        let customer_set: HashSet<usize> = visited_facts.iter().map(|v| v.place).collect();
        let depot = loc_facts
            .iter()
            .find(|loc| !customer_set.contains(&loc.place))
            .map(|loc| loc.place)
            .ok_or_else(|| HeuristicBuildError::from("no depot found".to_string()))?;
        let mut customers: Vec<usize> = customer_set.into_iter().collect();
        customers.sort_unstable();
        let n_places = names.len();

        let mut load_id = None;
        let mut vehicles_id = None;
        let mut capacity = INF;
        let mut max_vehicles = INF;
        let mut demands = vec![0.0; n_places];

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let var_name = var.name().to_string();
            let t = var_name.trim();
            if t == "load()" || t == "PNE load()" {
                load_id = Some(id);
            } else if t == "vehicles()" || t == "PNE vehicles()" {
                vehicles_id = Some(id);
            } else if t == "capacity()" || t == "PNE capacity()" {
                capacity = task
                    .get_initial_numeric_state_values()
                    .get(id)
                    .copied()
                    .unwrap_or(INF)
                    .max(0.0);
            } else if t == "max_vehicles()" || t == "PNE max_vehicles()" {
                max_vehicles = task
                    .get_initial_numeric_state_values()
                    .get(id)
                    .copied()
                    .unwrap_or(INF)
                    .max(0.0);
            }

            for p in 0..n_places {
                let obj = &names[p];
                let expected = format!("demand({obj})");
                let expected_pne = format!("PNE demand({obj})");
                if t == expected || t == expected_pne {
                    demands[p] = task
                        .get_initial_numeric_state_values()
                        .get(id)
                        .copied()
                        .unwrap_or(0.0)
                        .max(0.0);
                }
            }
        }

        let mut direct_costs = vec![vec![INF; n_places]; n_places];
        let mut via_depot_costs = vec![vec![INF; n_places]; n_places];
        for p in 0..n_places {
            direct_costs[p][p] = 0.0;
            via_depot_costs[p][p] = 0.0;
        }

        for op in task.get_operators() {
            let op_name = op.name().to_string();
            let parts: Vec<&str> = op_name.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let cost = metric_operator_cost_from_initial_values(task, op) as f64;
            if parts[0] == "visit" && parts.len() == 3 {
                let from = intern(parts[1], &mut name_to_id, &mut names);
                let to = intern(parts[2], &mut name_to_id, &mut names);
                if from < n_places && to < n_places {
                    direct_costs[from][to] = direct_costs[from][to].min(cost);
                }
            } else if parts[0] == "visit-via-depot" && parts.len() == 3 {
                let from = intern(parts[1], &mut name_to_id, &mut names);
                let to = intern(parts[2], &mut name_to_id, &mut names);
                if from < n_places && to < n_places {
                    via_depot_costs[from][to] = via_depot_costs[from][to].min(cost);
                }
            } else if parts[0] == "return-to-depot" && parts.len() == 2 {
                let from = intern(parts[1], &mut name_to_id, &mut names);
                if from < n_places {
                    direct_costs[from][depot] = direct_costs[from][depot].min(cost);
                }
            }
        }

        // Phase B: fill missing direct arcs from explicit travel-cost PNEs.
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let var_name = var.name().to_string();
            let t = var_name.trim();
            if let Some((from_name, to_name)) = parse_travel_cost_name(t) {
                if let (Some(&from), Some(&to)) = (name_to_id.get(&from_name), name_to_id.get(&to_name)) {
                    if from < n_places && to < n_places && direct_costs[from][to] >= INF {
                        let value = task
                            .get_initial_numeric_state_values()
                            .get(id)
                            .copied()
                            .unwrap_or(INF);
                        direct_costs[from][to] = value.max(0.0);
                    }
                }
            }
        }

        // Phase C: fill missing arcs via depot when both legs are known.
        for from in 0..n_places {
            for to in 0..n_places {
                if from != to && direct_costs[from][to] >= INF {
                    if direct_costs[from][depot] < INF && direct_costs[depot][to] < INF {
                        direct_costs[from][to] =
                            direct_costs[from][depot] + direct_costs[depot][to];
                    }
                }
            }
        }

        // Fill missing via-depot costs from direct depot legs.
        for &from in &customers {
            for &to in &customers {
                if from != to && via_depot_costs[from][to] >= INF {
                    if direct_costs[from][depot] < INF && direct_costs[depot][to] < INF {
                        via_depot_costs[from][to] =
                            direct_costs[from][depot] + direct_costs[depot][to];
                    }
                }
            }
        }

        let missing_depot_out = customers
            .iter()
            .filter(|&&c| direct_costs[depot][c] >= INF)
            .count();
        if missing_depot_out > 0 {
            return Err(format!(
                "unusable distance matrix: missing depot->customer costs for {} customers",
                missing_depot_out
            )
            .into());
        }

        let mut nearest_direct = vec![Vec::new(); n_places];
        for p in 0..n_places {
            let mut list: Vec<(usize, f64)> = customers
                .iter()
                .filter(|&&q| q != p && direct_costs[p][q] < INF)
                .map(|&q| (q, direct_costs[p][q]))
                .collect();
            list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            nearest_direct[p] = list;
        }

        let mut nearest_via = vec![Vec::new(); n_places];
        for &p in &customers {
            let mut list: Vec<(usize, f64)> = customers
                .iter()
                .filter(|&&q| q != p && via_depot_costs[p][q] < INF)
                .map(|&q| (q, via_depot_costs[p][q]))
                .collect();
            list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            nearest_via[p] = list;
        }

        let mut round_trip_sum = 0.0;
        let mut round_trip_count = 0usize;
        for &c in &customers {
            if direct_costs[depot][c] < INF && direct_costs[c][depot] < INF {
                round_trip_sum += direct_costs[depot][c] + direct_costs[c][depot];
                round_trip_count += 1;
            }
        }
        let avg_depot_round_trip = if round_trip_count > 0 {
            round_trip_sum / round_trip_count as f64
        } else {
            0.0
        };
        let vehicle_use_penalty = avg_depot_round_trip * 0.1;

        Ok(Self {
            n_places,
            depot,
            customers,
            loc_facts,
            visited_facts,
            numeric_var_count: task.numeric_variables().len(),
            prop_var_count: task.variables().len(),
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            direct_costs,
            via_depot_costs,
            nearest_direct,
            nearest_via,
            avg_depot_round_trip,
            vehicle_use_penalty,
            scratch_numeric: RefCell::new(Vec::new()),
            scratch_prop: RefCell::new(Vec::new()),
            scratch_visited: RefCell::new(Vec::new()),
        })
    }

    fn direct_cost(&self, from: usize, to: usize) -> f64 {
        self.direct_costs
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(INF)
    }

    fn via_depot_cost(&self, from: usize, to: usize) -> f64 {
        self.via_depot_costs
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(INF)
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
            numeric_values.clear();
            numeric_values.resize(self.numeric_var_count, 0.0);
            registry
                .view(state)
                .fill_numeric(&mut numeric_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
        }
        let load = {
            let numeric_values = self.scratch_numeric.borrow();
            self.load_id
                .and_then(|id| numeric_values.get(id))
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        };
        let vehicles = {
            let numeric_values = self.scratch_numeric.borrow();
            self.vehicles_id
                .and_then(|id| numeric_values.get(id))
                .copied()
                .unwrap_or(1.0)
                .max(0.0)
        };

        {
            let mut prop_values = self.scratch_prop.borrow_mut();
            prop_values.clear();
            prop_values.resize(self.prop_var_count, 0usize);
            state
                .fill_state(registry, &mut prop_values)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
        }
        let prop_values = self.scratch_prop.borrow();

        let mut current = self.depot;
        for loc in &self.loc_facts {
            if prop_values.get(loc.var_id).copied() == Some(loc.value) {
                current = loc.place;
                break;
            }
        }

        let mut visited = self.scratch_visited.borrow_mut();
        visited.clear();
        visited.resize(self.n_places, false);
        for v in &self.visited_facts {
            if prop_values.get(v.var_id).copied() == Some(v.value) {
                visited[v.place] = true;
            }
        }

        let mut unvisited_count = 0usize;
        let mut remaining_demand = 0.0;
        for &c in &self.customers {
            if !visited[c] {
                unvisited_count += 1;
                remaining_demand += self.demands[c];
            }
        }

        if unvisited_count == 0 {
            if current != self.depot {
                let c = self.direct_cost(current, self.depot);
                return Ok(if c < INF { c } else { 0.0 });
            }
            return Ok(0.0);
        }

        // Resource infeasibility check.
        let residual_cap = (self.capacity - load).max(0.0);
        let rem_vehicles = if self.max_vehicles >= INF / 2.0 {
            INF
        } else {
            (self.max_vehicles - vehicles).max(0.0)
        };
        let available_cap = if self.capacity >= INF / 2.0 || rem_vehicles >= INF / 2.0 {
            INF
        } else {
            residual_cap + rem_vehicles * self.capacity.max(0.0)
        };
        if remaining_demand > available_cap + 1e-6 {
            return Ok(LARGE_PENALTY + remaining_demand);
        }

        // Capacity pressure component: rough extra depot reload cost.
        let estimated_additional_via = if self.capacity > 1.0 && remaining_demand > residual_cap {
            ((remaining_demand - residual_cap) / self.capacity).ceil()
        } else {
            0.0
        };
        let h_capacity = estimated_additional_via * self.avg_depot_round_trip * 0.35;

        let h_vehicles = if self.max_vehicles < INF / 2.0 {
            let used_ratio = (vehicles / self.max_vehicles.max(1.0)).clamp(0.0, 1.0);
            used_ratio * self.avg_depot_round_trip * 0.15
        } else {
            0.0
        };

        // State-dependent greedy routing completion.
        let mut h_routing = 0.0;
        let mut cur = current;
        let mut cur_load = load;
        let mut cur_vehicles = vehicles;
        let mut remaining = unvisited_count;
        let mut dead_end = false;

        while remaining > 0 {
            let mut best_action = NextAction::Direct;
            let mut best_candidate: Option<usize> = None;
            let mut best_cost = INF;

            if let Some(list) = self.nearest_direct.get(cur) {
                for &(candidate, edge_cost) in list {
                    if visited[candidate] {
                        continue;
                    }
                    if self.capacity >= INF / 2.0
                        || cur_load + self.demands[candidate] <= self.capacity + 1e-9
                    {
                        best_action = NextAction::Direct;
                        best_candidate = Some(candidate);
                        best_cost = edge_cost;
                        break;
                    }
                }
            }

            if cur != self.depot
                && (self.max_vehicles >= INF / 2.0 || cur_vehicles < self.max_vehicles)
            {
                if let Some(via_list) = self.nearest_via.get(cur) {
                    for &(candidate, edge_cost) in via_list {
                        if visited[candidate] {
                            continue;
                        }
                        let total = edge_cost + self.vehicle_use_penalty;
                        if total < best_cost {
                            best_action = NextAction::ViaDepot;
                            best_candidate = Some(candidate);
                            best_cost = total;
                        }
                        break;
                    }
                }
            }

            if let Some(candidate) = best_candidate {
                visited[candidate] = true;
                remaining -= 1;
                match best_action {
                    NextAction::Direct => {
                        h_routing += self.direct_cost(cur, candidate);
                        cur_load += self.demands[candidate];
                    }
                    NextAction::ViaDepot => {
                        h_routing += self.via_depot_cost(cur, candidate) + self.vehicle_use_penalty;
                        cur_load = self.demands[candidate];
                        cur_vehicles += 1.0;
                    }
                }
                cur = candidate;
            } else {
                dead_end = true;
                break;
            }
        }

        if dead_end {
            return Ok(LARGE_PENALTY + remaining as f64 + remaining_demand);
        }

        if cur != self.depot {
            let ret = self.direct_cost(cur, self.depot);
            h_routing += if ret < INF { ret } else { 0.0 };
        }

        let h = h_routing + h_capacity + h_vehicles;
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
