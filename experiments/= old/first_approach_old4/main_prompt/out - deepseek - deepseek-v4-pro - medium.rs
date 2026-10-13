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
use std::collections::{HashMap, HashSet};

fn extract_object(text: &str, needle: &str) -> Option<String> {
    let start = text.find(needle)? + needle.len();
    let end = text[start..].find(')')? + start;
    Some(text[start..end].to_string())
}

fn intern_place(
    name: &str,
    place_names: &mut Vec<String>,
    name_to_idx: &mut HashMap<String, usize>,
    loc_candidates: &mut Vec<Vec<(usize, usize)>>,
    visited_candidates: &mut Vec<Vec<(usize, usize)>>,
) -> usize {
    if let Some(&idx) = name_to_idx.get(name) {
        return idx;
    }
    let idx = place_names.len();
    place_names.push(name.to_string());
    name_to_idx.insert(name.to_string(), idx);
    loc_candidates.push(Vec::new());
    visited_candidates.push(Vec::new());
    idx
}

struct LlmHeuristic {
    n_places: usize,
    depot_idx: usize,
    customer_indices: Vec<usize>,
    dist: Vec<f64>,
    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    loc_candidates: Vec<Vec<(usize, usize)>>,
    visited_candidates: Vec<Vec<(usize, usize)>>,
    num_prop_vars: usize,
    num_numeric_vars: usize,
    numeric_scratch: RefCell<Vec<f64>>,
    prop_scratch: RefCell<Vec<usize>>,
    visited_scratch: RefCell<Vec<bool>>,
    unvisited_scratch: RefCell<Vec<usize>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let num_prop_vars = task.variables().len();
        let num_numeric_vars = task.numeric_variables().len();

        let mut place_names: Vec<String> = Vec::new();
        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        let mut loc_candidates: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut visited_candidates: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut loc_objects: HashSet<String> = HashSet::new();
        let mut visited_objects: HashSet<String> = HashSet::new();

        let variables = task.variables();
        for var_id in 0..variables.len() {
            let domain_size = variables[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();

                if let Some(obj) = extract_object(&fact_name, "visited(") {
                    let idx = intern_place(
                        &obj,
                        &mut place_names,
                        &mut name_to_idx,
                        &mut loc_candidates,
                        &mut visited_candidates,
                    );
                    visited_candidates[idx].push((var_id, value));
                    visited_objects.insert(obj);
                }

                if let Some(obj) = extract_object(&fact_name, "loc(") {
                    let idx = intern_place(
                        &obj,
                        &mut place_names,
                        &mut name_to_idx,
                        &mut loc_candidates,
                        &mut visited_candidates,
                    );
                    loc_candidates[idx].push((var_id, value));
                    loc_objects.insert(obj);
                }
            }
        }

        let depot_idx = if let Some(name) = loc_objects
            .iter()
            .find(|n| !visited_objects.contains(*n))
            .cloned()
        {
            intern_place(
                &name,
                &mut place_names,
                &mut name_to_idx,
                &mut loc_candidates,
                &mut visited_candidates,
            )
        } else if let Some(name) = loc_objects.iter().next().cloned() {
            intern_place(
                &name,
                &mut place_names,
                &mut name_to_idx,
                &mut loc_candidates,
                &mut visited_candidates,
            )
        } else {
            0
        };

        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for op in task.get_operators() {
            let name = op.name().to_string();
            let cost = metric_operator_cost_from_initial_values(task, op);
            let tokens: Vec<&str> = name.split_whitespace().collect();

            match tokens.first().copied() {
                Some("visit") => {
                    if tokens.len() >= 3 {
                        let from = intern_place(
                            tokens[1],
                            &mut place_names,
                            &mut name_to_idx,
                            &mut loc_candidates,
                            &mut visited_candidates,
                        );
                        let to = intern_place(
                            tokens[2],
                            &mut place_names,
                            &mut name_to_idx,
                            &mut loc_candidates,
                            &mut visited_candidates,
                        );
                        edges.push((from, to, cost));
                    }
                }
                Some("visit-via-depot") => {
                    if tokens.len() >= 3 {
                        intern_place(
                            tokens[1],
                            &mut place_names,
                            &mut name_to_idx,
                            &mut loc_candidates,
                            &mut visited_candidates,
                        );
                        intern_place(
                            tokens[2],
                            &mut place_names,
                            &mut name_to_idx,
                            &mut loc_candidates,
                            &mut visited_candidates,
                        );
                    }
                }
                Some("return-to-depot") => {
                    if tokens.len() >= 2 {
                        let from = intern_place(
                            tokens[1],
                            &mut place_names,
                            &mut name_to_idx,
                            &mut loc_candidates,
                            &mut visited_candidates,
                        );
                        edges.push((from, depot_idx, cost));
                    }
                }
                _ => {}
            }
        }

        let customer_indices: Vec<usize> = (0..place_names.len())
            .filter(|&i| !visited_candidates[i].is_empty())
            .collect();

        let n = place_names.len();
        let mut dist = vec![f64::INFINITY; n * n];
        for i in 0..n {
            dist[i * n + i] = 0.0;
        }

        for (a, b, c) in edges {
            let slot = &mut dist[a * n + b];
            if c < *slot {
                *slot = c;
            }
        }

        for i in 0..n {
            for j in 0..n {
                if i != j && !dist[i * n + j].is_finite() && dist[j * n + i].is_finite() {
                    dist[i * n + j] = dist[j * n + i];
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                if i != j
                    && !dist[i * n + j].is_finite()
                    && dist[i * n + depot_idx].is_finite()
                    && dist[depot_idx * n + j].is_finite()
                {
                    dist[i * n + j] = dist[i * n + depot_idx] + dist[depot_idx * n + j];
                }
            }
        }

        for v in dist.iter_mut() {
            if !v.is_finite() {
                *v = 1e9;
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut load_var = None;
        let mut vehicles_var = None;
        let mut max_vehicles = f64::INFINITY;
        let mut capacity = f64::INFINITY;
        let mut demands = vec![0.0; n];

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            let lower = name.to_ascii_lowercase();
            let initial_val = initial_numeric.get(id).copied().unwrap_or(0.0);

            if lower.contains("load()") || lower == "load" {
                if load_var.is_none() {
                    load_var = Some(id);
                }
            } else if lower.contains("max_vehicles") {
                if initial_val > 0.0 {
                    max_vehicles = initial_val;
                }
            } else if lower.contains("vehicles") {
                if vehicles_var.is_none() {
                    vehicles_var = Some(id);
                }
            } else if lower.contains("capacity") {
                if initial_val > 0.0 {
                    capacity = initial_val;
                }
            } else if lower.contains("demand") {
                if let Some(obj) = extract_object(&name, "demand(") {
                    if let Some(&idx) = name_to_idx.get(&obj) {
                        demands[idx] = initial_val;
                    }
                }
            }
        }

        Ok(Self {
            n_places: n,
            depot_idx,
            customer_indices,
            dist,
            demands,
            capacity,
            max_vehicles,
            load_var,
            vehicles_var,
            loc_candidates,
            visited_candidates,
            num_prop_vars,
            num_numeric_vars,
            numeric_scratch: RefCell::new(vec![0.0; num_numeric_vars]),
            prop_scratch: RefCell::new(vec![0; num_prop_vars]),
            visited_scratch: RefCell::new(Vec::new()),
            unvisited_scratch: RefCell::new(Vec::new()),
        })
    }

    #[inline]
    fn distance(&self, a: usize, b: usize) -> f64 {
        self.dist[a * self.n_places + b]
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
            let mut numeric_guard = self.numeric_scratch.borrow_mut();
            let numeric_values = &mut *numeric_guard;
            if numeric_values.len() != self.num_numeric_vars {
                numeric_values.resize(self.num_numeric_vars, 0.0);
            }
            registry
                .view(state)
                .fill_numeric(numeric_values)
                .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

            let load = self.load_var.map(|id| numeric_values[id]).unwrap_or(0.0);
            let vehicles = self
                .vehicles_var
                .map(|id| numeric_values[id])
                .unwrap_or(1.0)
                .max(1.0);

            (load, vehicles)
        };

        let mut current_place = self.depot_idx;
        {
            let mut prop_guard = self.prop_scratch.borrow_mut();
            let prop_values = &mut *prop_guard;
            if prop_values.len() != self.num_prop_vars {
                prop_values.resize(self.num_prop_vars, 0);
            }
            state
                .fill_state(registry, prop_values)
                .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

            for (place_idx, candidates) in self.loc_candidates.iter().enumerate() {
                if candidates
                    .iter()
                    .any(|(vid, val)| prop_values[*vid] == *val)
                {
                    current_place = place_idx;
                    break;
                }
            }

            let mut visited_guard = self.visited_scratch.borrow_mut();
            visited_guard.clear();
            visited_guard.resize(self.n_places, false);

            let mut unvisited_guard = self.unvisited_scratch.borrow_mut();
            unvisited_guard.clear();

            for &cust in &self.customer_indices {
                let is_visited = self.visited_candidates[cust]
                    .iter()
                    .any(|(vid, val)| prop_values[*vid] == *val);
                visited_guard[cust] = is_visited;
                if !is_visited {
                    unvisited_guard.push(cust);
                }
            }
        }

        let mut unvisited_guard = self.unvisited_scratch.borrow_mut();
        let unvisited = &mut *unvisited_guard;

        if unvisited.is_empty() {
            if current_place != self.depot_idx {
                return Ok(self.distance(current_place, self.depot_idx));
            }
            return Ok(0.0);
        }

        let mut total = 0.0;
        let mut last = current_place;
        let mut current_load = load;
        let mut routes_used = vehicles;

        while !unvisited.is_empty() {
            let mut best_pos: Option<usize> = None;
            let mut best_cost = f64::INFINITY;
            let mut best_via = false;

            for (pos, &v) in unvisited.iter().enumerate() {
                let direct_ok = if self.capacity.is_finite() {
                    current_load + self.demands[v] <= self.capacity + 1e-9
                } else {
                    true
                };

                if direct_ok {
                    let d = self.distance(last, v);
                    if d.is_finite() && d < best_cost {
                        best_cost = d;
                        best_pos = Some(pos);
                        best_via = false;
                    }
                }

                let via_ok = last != self.depot_idx
                    && (self.max_vehicles.is_infinite()
                        || routes_used < self.max_vehicles - 1e-9);
                if via_ok {
                    let d = self.distance(last, self.depot_idx)
                        + self.distance(self.depot_idx, v);
                    if d.is_finite() && d < best_cost {
                        best_cost = d;
                        best_pos = Some(pos);
                        best_via = true;
                    }
                }
            }

            if let Some(pos) = best_pos {
                let v = unvisited.swap_remove(pos);
                if best_via {
                    total += self.distance(last, self.depot_idx)
                        + self.distance(self.depot_idx, v);
                    routes_used += 1.0;
                    current_load = self.demands[v];
                } else {
                    total += self.distance(last, v);
                    current_load += self.demands[v];
                }
                last = v;
            } else {
                for &v in unvisited.iter() {
                    total += self.distance(self.depot_idx, v)
                        + self.distance(v, self.depot_idx)
                        + 1_000_000.0;
                }
                break;
            }
        }

        if last != self.depot_idx {
            total += self.distance(last, self.depot_idx);
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
