use std::cell::RefCell;
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

const INF: f64 = f64::INFINITY;
const LARGE_PENALTY: f64 = 1e12;

fn parse_pred_object(name: &str, pred: &str) -> Option<String> {
    let marker = format!("Atom {pred}(");
    if let Some(pos) = name.find(&marker) {
        let start = pos + marker.len();
        let end = name[start..].find(')')? + start;
        return Some(name[start..end].to_string());
    }
    let marker2 = format!("{pred}(");
    if let Some(pos) = name.find(&marker2) {
        let start = pos + marker2.len();
        let end = name[start..].find(')')? + start;
        return Some(name[start..end].to_string());
    }
    None
}

fn parse_two_tokens(s: &str) -> Option<(&str, &str)> {
    let mut it = s.split_whitespace();
    let a = it.next()?;
    let b = it.next()?;
    if it.next().is_some() {
        None
    } else {
        Some((a, b))
    }
}

fn parse_one_token(s: &str) -> Option<&str> {
    let mut it = s.split_whitespace();
    let a = it.next()?;
    if it.next().is_some() {
        None
    } else {
        Some(a)
    }
}

fn parse_travel_cost(name: &str) -> Option<(&str, &str)> {
    let marker = "travel-cost(";
    let pos = name.find(marker)?;
    let start = pos + marker.len();
    let end = name[start..].find(')')? + start;
    let inner = &name[start..end];
    let mut it = inner.split(|c: char| c.is_whitespace() || c == ',');
    let a = it.next()?;
    let b = it.next()?;
    if it.next().is_some() {
        return None;
    }
    Some((a, b))
}

struct LlmHeuristic {
    node_count: usize,
    num_numeric_vars: usize,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: Vec<f64>,
    cost_direct: Vec<Vec<f64>>,
    cost_via: Vec<Vec<f64>>,
    cost_to_depot: Vec<f64>,
    nearest_direct: Vec<Vec<usize>>,
    nearest_via: Vec<Vec<usize>>,
    loc_facts: Vec<(ExplicitFact, usize)>,
    visited_facts: Vec<(ExplicitFact, usize)>,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_flags: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_facts: Vec<(ExplicitFact, String)> = Vec::new();
        let mut visited_facts: Vec<(ExplicitFact, String)> = Vec::new();

        for var_id in 0..task.variables().len() {
            for value in 0..task.variables()[var_id].domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact).to_string();
                if let Some(obj) = parse_pred_object(&name, "loc") {
                    loc_facts.push((fact, obj));
                } else if let Some(obj) = parse_pred_object(&name, "visited") {
                    visited_facts.push((fact, obj));
                }
            }
        }

        let visited_names: HashSet<String> = visited_facts.iter().map(|(_, n)| n.clone()).collect();
        let depot_name = loc_facts
            .iter()
            .map(|(_, n)| n.clone())
            .find(|n| !visited_names.contains(n))
            .ok_or_else(|| HeuristicBuildError::from("missing depot loc fact".to_string()))?;

        let mut customer_names: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (_, n) in &visited_facts {
            if seen.insert(n.clone()) {
                customer_names.push(n.clone());
            }
        }

        let m = customer_names.len() + 1;
        let mut place_to_idx: HashMap<String, usize> = HashMap::new();
        place_to_idx.insert(depot_name.clone(), 0);
        for (i, n) in customer_names.iter().enumerate() {
            place_to_idx.insert(n.clone(), i + 1);
        }

        let loc_idx_facts: Vec<(ExplicitFact, usize)> = loc_facts
            .into_iter()
            .filter_map(|(fact, obj)| place_to_idx.get(&obj).map(|&idx| (fact, idx)))
            .collect();
        let visited_idx_facts: Vec<(ExplicitFact, usize)> = visited_facts
            .into_iter()
            .filter_map(|(fact, obj)| place_to_idx.get(&obj).map(|&idx| (fact, idx)))
            .collect();

        let num_numeric_vars = task.numeric_variables().len();
        let mut load_id = None;
        let mut vehicles_id = None;
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let n = var.name().trim();
            if n == "load()" || n == "PNE load()" {
                load_id = Some(id);
            } else if n == "vehicles()" || n == "PNE vehicles()" {
                vehicles_id = Some(id);
            }
        }
        let load_id = load_id.ok_or_else(|| HeuristicBuildError::from("missing load numeric variable".to_string()))?;
        let vehicles_id = vehicles_id.ok_or_else(|| HeuristicBuildError::from("missing vehicles numeric variable".to_string()))?;

        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let n = var.name().trim();
            if n == "capacity()" || n == "PNE capacity()" {
                capacity = task.get_initial_numeric_state_values()[id];
            } else if n == "max_vehicles()" || n == "PNE max_vehicles()" {
                max_vehicles = task.get_initial_numeric_state_values()[id];
            }
        }

        let mut demands = vec![0.0; m];
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let n = var.name().trim();
            for (i, cname) in customer_names.iter().enumerate() {
                if n == format!("demand({})", cname) || n == format!("PNE demand({})", cname) {
                    demands[i + 1] = task.get_initial_numeric_state_values()[id];
                    break;
                }
            }
        }

        let mut cost_direct = vec![vec![INF; m]; m];
        let mut cost_via = vec![vec![INF; m]; m];
        let mut cost_to_depot = vec![INF; m];

        for op in task.get_operators() {
            let name = op.name().to_string();
            let cost = metric_operator_cost_from_initial_values(task, op);
            if let Some(rest) = name.strip_prefix("visit-via-depot ") {
                if let Some((a, b)) = parse_two_tokens(rest) {
                    if let (Some(&i), Some(&j)) = (place_to_idx.get(a), place_to_idx.get(b)) {
                        if i != j {
                            cost_via[i][j] = cost;
                        }
                    }
                }
            } else if let Some(rest) = name.strip_prefix("visit ") {
                if let Some((a, b)) = parse_two_tokens(rest) {
                    if let (Some(&i), Some(&j)) = (place_to_idx.get(a), place_to_idx.get(b)) {
                        cost_direct[i][j] = cost;
                    }
                }
            } else if let Some(rest) = name.strip_prefix("return-to-depot ") {
                if let Some(a) = parse_one_token(rest) {
                    if let Some(&i) = place_to_idx.get(a) {
                        cost_to_depot[i] = cost;
                        cost_direct[i][0] = cost;
                    }
                }
            }
        }

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name();
            if name.contains("travel-cost(") {
                if let Some((a, b)) = parse_travel_cost(name) {
                    if let (Some(&i), Some(&j)) = (place_to_idx.get(a), place_to_idx.get(b)) {
                        if !cost_direct[i][j].is_finite() {
                            cost_direct[i][j] = task.get_initial_numeric_state_values()[id];
                        }
                    }
                }
            }
        }

        for i in 1..m {
            for j in 1..m {
                if i != j
                    && !cost_direct[i][j].is_finite()
                    && cost_to_depot[i].is_finite()
                    && cost_direct[0][j].is_finite()
                {
                    cost_direct[i][j] = cost_to_depot[i] + cost_direct[0][j];
                }
            }
        }

        if m > 1 {
            let has_depot_out = (1..m).any(|j| cost_direct[0][j].is_finite());
            let has_depot_in = (1..m).any(|i| cost_to_depot[i].is_finite());
            if !has_depot_out || !has_depot_in {
                return Err(HeuristicBuildError::from(
                    "unusable travel cost matrix: missing depot connections".to_string(),
                ));
            }
        }

        let mut nearest_direct = Vec::with_capacity(m);
        for i in 0..m {
            let mut list: Vec<usize> = (1..m).filter(|j| *j != i).collect();
            list.sort_by(|a, b| {
                cost_direct[i][*a]
                    .partial_cmp(&cost_direct[i][*b])
                    .unwrap_or(Ordering::Equal)
            });
            nearest_direct.push(list);
        }

        let mut nearest_via = Vec::with_capacity(m);
        for i in 0..m {
            if i == 0 {
                nearest_via.push(Vec::new());
            } else {
                let mut list: Vec<usize> = (1..m).filter(|j| *j != i).collect();
                list.sort_by(|a, b| {
                    cost_via[i][*a]
                        .partial_cmp(&cost_via[i][*b])
                        .unwrap_or(Ordering::Equal)
                });
                nearest_via.push(list);
            }
        }

        Ok(Self {
            node_count: m,
            num_numeric_vars,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demands,
            cost_direct,
            cost_via,
            cost_to_depot,
            nearest_direct,
            nearest_via,
            loc_facts: loc_idx_facts,
            visited_facts: visited_idx_facts,
            scratch_numeric: RefCell::new(Vec::with_capacity(num_numeric_vars)),
            scratch_flags: RefCell::new(Vec::with_capacity(m)),
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

        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let (load, vehicles) = {
            let mut nv = self.scratch_numeric.borrow_mut();
            nv.clear();
            nv.resize(self.num_numeric_vars, 0.0);
            registry
                .view(state)
                .fill_numeric(&mut nv)
                .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;
            (nv[self.load_id], nv[self.vehicles_id])
        };

        let mut flags = self.scratch_flags.borrow_mut();
        flags.clear();
        flags.resize(self.node_count, false);
        for (fact, idx) in &self.visited_facts {
            if fact.is_hold(registry.view(state)) {
                flags[*idx] = true;
            }
        }

        let mut loc_idx: Option<usize> = None;
        for (fact, idx) in &self.loc_facts {
            if fact.is_hold(registry.view(state)) {
                loc_idx = Some(*idx);
                break;
            }
        }
        let loc_idx = loc_idx.ok_or_else(|| {
            EvaluationError::ComputationFailed("no held loc fact".to_string())
        })?;

        let mut unvisited = 0usize;
        let mut total_demand = 0.0_f64;
        for idx in 1..self.node_count {
            if !flags[idx] {
                unvisited += 1;
                total_demand += self.demands[idx];
            }
        }

        let available_cap = {
            if self.capacity.is_finite() && self.max_vehicles.is_finite() {
                (self.capacity - load).max(0.0)
                    + (self.max_vehicles - vehicles).max(0.0) * self.capacity
            } else {
                f64::MAX
            }
        };

        if self.capacity.is_finite() && total_demand > available_cap + 1e-9 {
            let excess = total_demand - available_cap;
            return Ok(LARGE_PENALTY + excess.max(0.0) * 100.0);
        }

        if unvisited == 0 {
            if loc_idx == 0 {
                return Ok(0.0);
            }
            let c = self.cost_to_depot[loc_idx];
            return if c.is_finite() {
                Ok(c)
            } else {
                Ok(LARGE_PENALTY)
            };
        }

        let mut curr = loc_idx;
        let mut cur_load = load;
        let mut cur_veh = vehicles;
        let mut remaining = unvisited;
        let mut h_routing = 0.0_f64;
        let mut failed = false;

        while remaining > 0 {
            let mut best_node: Option<usize> = None;
            let mut best_cost = f64::INFINITY;
            let mut best_via = false;

            for &cand in &self.nearest_direct[curr] {
                if flags[cand] {
                    continue;
                }
                let fits = !self.capacity.is_finite()
                    || cur_load + self.demands[cand] <= self.capacity + 1e-9;
                if fits {
                    let c = self.cost_direct[curr][cand];
                    if c.is_finite() && c < best_cost {
                        best_cost = c;
                        best_node = Some(cand);
                        best_via = false;
                    }
                    break;
                }
            }

            let via_allowed = curr != 0
                && (!self.max_vehicles.is_finite() || cur_veh < self.max_vehicles);
            if via_allowed {
                for &cand in &self.nearest_via[curr] {
                    if flags[cand] {
                        continue;
                    }
                    let fits = !self.capacity.is_finite()
                        || self.demands[cand] <= self.capacity + 1e-9;
                    if fits {
                        let c = self.cost_via[curr][cand];
                        if c.is_finite() && c < best_cost {
                            best_cost = c;
                            best_node = Some(cand);
                            best_via = true;
                        }
                        break;
                    }
                }
            }

            match best_node {
                Some(next) => {
                    h_routing += best_cost;
                    if best_via {
                        cur_load = self.demands[next];
                        cur_veh += 1.0;
                    } else {
                        cur_load += self.demands[next];
                    }
                    curr = next;
                    flags[next] = true;
                    remaining -= 1;
                }
                None => {
                    failed = true;
                    break;
                }
            }
        }

        if failed {
            return Ok(LARGE_PENALTY + remaining as f64 * 100.0 + h_routing);
        }

        if curr != 0 {
            let c = self.cost_to_depot[curr];
            if c.is_finite() {
                h_routing += c;
            } else {
                return Ok(LARGE_PENALTY + h_routing);
            }
        }

        let h_capacity = if self.capacity.is_finite() && self.capacity > 0.0 {
            total_demand / self.capacity
        } else {
            0.0
        };

        Ok(h_routing + h_capacity)
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
        return Err("`llm_001` does not accept arguments"
            .to_string()
            .into());
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
