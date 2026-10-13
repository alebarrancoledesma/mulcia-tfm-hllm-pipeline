use std::cell::RefCell;
use std::collections::HashMap;

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::task::ExplicitFact;
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

fn parse_fact_name(name: &str) -> Option<(&str, Vec<&str>)> {
    let inner = name
        .strip_prefix("Atom ")
        .or_else(|| name.strip_prefix("NegatedAtom "))
        .unwrap_or(name);
    let open = inner.find('(')?;
    let close = inner.rfind(')')?;
    let pred = &inner[..open];
    let args_str = &inner[open + 1..close];
    let args = args_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    Some((pred, args))
}

struct LlmHeuristic {
    n: usize,
    depot: usize,
    visited_facts: Vec<ExplicitFact>,
    loc_facts: Vec<Option<ExplicitFact>>,
    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    dist: Vec<Vec<f64>>,
    nearest_from: Vec<Vec<usize>>,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_visited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_entries: Vec<(String, ExplicitFact)> = Vec::new();
        let mut visited_entries: Vec<(String, ExplicitFact)> = Vec::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in var.values().iter() {
                let value_idx = *value as usize;
                let fact = ExplicitFact::propositional(var_id, value_idx);
                let fact_name = task.get_fact_name(fact);
                if let Some((pred, args)) = parse_fact_name(&fact_name) {
                    match pred {
                        "loc" if args.len() == 1 => {
                            loc_entries.push((args[0].to_string(), fact));
                        }
                        "visited" if args.len() == 1 => {
                            visited_entries.push((args[0].to_string(), fact));
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut customer_names: Vec<String> = visited_entries
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        customer_names.sort();
        customer_names.dedup();
        let n = customer_names.len();
        if n == 0 {
            return Err("llm_001: no visited customers found".to_string().into());
        }

        let mut name_to_idx: HashMap<String, usize> = customer_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut depot_names: Vec<String> = loc_entries
            .iter()
            .map(|(name, _)| name.clone())
            .filter(|name| !name_to_idx.contains_key(name))
            .collect();
        depot_names.sort();
        depot_names.dedup();
        let depot_name = match depot_names.first() {
            Some(name) => name.clone(),
            None => return Err("llm_001: no depot found".to_string().into()),
        };
        name_to_idx.insert(depot_name, n);
        let depot = n;

        let mut visited_facts: Vec<Option<ExplicitFact>> = vec![None; n];
        for (name, fact) in visited_entries {
            if let Some(&idx) = name_to_idx.get(&name) {
                if idx < n {
                    visited_facts[idx] = Some(fact);
                }
            }
        }
        let visited_facts: Vec<ExplicitFact> = visited_facts
            .into_iter()
            .map(|f| f.expect("llm_001: missing visited fact"))
            .collect();

        let mut loc_facts: Vec<Option<ExplicitFact>> = vec![None; n + 1];
        for (name, fact) in loc_entries {
            if let Some(&idx) = name_to_idx.get(&name) {
                loc_facts[idx] = Some(fact);
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let num_numeric = task.numeric_variables().len();
        let mut demands = vec![0.0; n];
        let mut capacity = f64::INFINITY;
        let mut max_vehicles = f64::INFINITY;
        let mut load_var = None;
        let mut vehicles_var = None;
        let mut dist = vec![vec![f64::INFINITY; n + 1]; n + 1];
        for i in 0..=n {
            dist[i][i] = 0.0;
        }

        for (num_id, nvar) in task.numeric_variables().iter().enumerate() {
            if num_id >= initial_numeric.len() {
                continue;
            }
            let val = initial_numeric[num_id];
            let name_string = nvar.name().to_string();
            let name = name_string.trim();

            if name.ends_with("max_vehicles()") || name.contains("max-vehicles()") {
                max_vehicles = val;
            } else if name.ends_with("vehicles()") {
                if vehicles_var.is_none() {
                    vehicles_var = Some(num_id);
                }
            }

            if name.ends_with("load()") {
                if load_var.is_none() {
                    load_var = Some(num_id);
                }
            }

            if name.ends_with("capacity()") {
                capacity = val;
            }

            if let Some(pos) = name.find("demand(") {
                let after = &name[pos + "demand(".len()..];
                if let Some(end) = after.find(')') {
                    let obj = after[..end].trim();
                    if let Some(&idx) = name_to_idx.get(obj) {
                        if idx < n {
                            demands[idx] = val;
                        }
                    }
                }
            }

            if let Some(pos) = name.find("travel-cost(") {
                let after = &name[pos + "travel-cost(".len()..];
                if let Some(end) = after.find(')') {
                    let args_str = &after[..end];
                    let args: Vec<&str> = args_str
                        .split(|c: char| c == ',' || c.is_whitespace())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if args.len() >= 2 {
                        if let (Some(&a), Some(&b)) = (
                            name_to_idx.get(args[0]),
                            name_to_idx.get(args[1]),
                        ) {
                            dist[a][b] = val;
                        }
                    }
                }
            }
        }

        for i in 0..=n {
            for j in 0..=n {
                if dist[i][j].is_infinite() && !dist[j][i].is_infinite() {
                    dist[i][j] = dist[j][i];
                }
            }
        }

        for i in 0..=n {
            for j in 0..=n {
                if i != depot && j != depot && dist[i][j].is_infinite() {
                    let via = dist[i][depot] + dist[depot][j];
                    if via.is_finite() {
                        dist[i][j] = via;
                    }
                }
            }
        }

        for i in 0..=n {
            for j in 0..=n {
                if dist[i][j].is_infinite() {
                    dist[i][j] = 1e9;
                }
            }
        }

        let mut nearest_from = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let mut nodes: Vec<usize> = (0..=n).filter(|&j| j != i).collect();
            nodes.sort_by(|&a, &b| {
                dist[i][a]
                    .partial_cmp(&dist[i][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
            nearest_from.push(nodes);
        }

        Ok(Self {
            n,
            depot,
            visited_facts,
            loc_facts,
            demands,
            capacity,
            max_vehicles,
            load_var,
            vehicles_var,
            dist,
            nearest_from,
            scratch_numeric: RefCell::new(vec![0.0; num_numeric]),
            scratch_visited: RefCell::new(vec![false; n]),
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
        let view = registry.view(state);

        let mut cur = self.depot;
        for i in 0..=self.n {
            if let Some(fact) = &self.loc_facts[i] {
                if fact.is_hold(view) {
                    cur = i;
                    break;
                }
            }
        }

        let mut flags = self.scratch_visited.borrow_mut();
        if flags.len() != self.n {
            flags.clear();
            flags.resize(self.n, false);
        }
        let mut remaining = 0usize;
        for i in 0..self.n {
            let unvisited = !self.visited_facts[i].is_hold(view);
            flags[i] = unvisited;
            if unvisited {
                remaining += 1;
            }
        }

        if remaining == 0 {
            return if cur == self.depot {
                Ok(0.0)
            } else {
                Ok(self.dist[cur][self.depot])
            };
        }

        let (mut cur_load, mut used_vehicles) = {
            let mut numeric_values = self.scratch_numeric.borrow_mut();
            registry.view(state).fill_numeric(&mut *numeric_values);
            let load = self.load_var.map(|id| numeric_values[id]);
            let veh = self.vehicles_var.map(|id| numeric_values[id]);
            let cur_load = if cur == self.depot {
                0.0
            } else {
                load.unwrap_or_else(|| self.demands[cur])
            };
            let used_vehicles = veh.unwrap_or(1.0).max(1.0);
            (cur_load, used_vehicles)
        };

        let mut total = 0.0;
        let cap = self.capacity;
        let max_v = self.max_vehicles;

        while remaining > 0 {
            let mut best_direct = None;
            for &cand in &self.nearest_from[cur] {
                if cand == self.depot {
                    continue;
                }
                if !flags[cand] {
                    continue;
                }
                if cur_load + self.demands[cand] <= cap + 1e-9 {
                    best_direct = Some(cand);
                    break;
                }
            }

            let mut best_via = None;
            if cur != self.depot && used_vehicles + 1.0 <= max_v + 1e-9 {
                for &cand in &self.nearest_from[self.depot] {
                    if cand == self.depot {
                        continue;
                    }
                    if !flags[cand] {
                        continue;
                    }
                    if self.demands[cand] <= cap + 1e-9 {
                        best_via = Some(cand);
                        break;
                    }
                }
            }

            if best_direct.is_none() && best_via.is_none() {
                return Ok(1e12);
            }

            let use_via = match (&best_direct, &best_via) {
                (Some(d), Some(v)) => {
                    let cost_direct = self.dist[cur][*d];
                    let cost_via = self.dist[cur][self.depot] + self.dist[self.depot][*v];
                    cost_via < cost_direct
                }
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (None, None) => unreachable!(),
            };

            let cand = if use_via {
                best_via.unwrap()
            } else {
                best_direct.unwrap()
            };
            let cost = if use_via {
                self.dist[cur][self.depot] + self.dist[self.depot][cand]
            } else {
                self.dist[cur][cand]
            };

            total += cost;
            flags[cand] = false;
            remaining -= 1;

            if use_via {
                used_vehicles += 1.0;
                cur_load = self.demands[cand];
            } else {
                if cur == self.depot {
                    cur_load = self.demands[cand];
                } else {
                    cur_load += self.demands[cand];
                }
            }
            cur = cand;
        }

        if cur != self.depot {
            total += self.dist[cur][self.depot];
        }

        Ok(total)
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
