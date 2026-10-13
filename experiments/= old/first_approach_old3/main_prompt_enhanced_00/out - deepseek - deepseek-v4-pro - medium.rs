use std::cell::RefCell;
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

const INF: f64 = 1e100;
const EPS: f64 = 1e-9;

fn extract_arg(name: &str, predicate: &str) -> Option<String> {
    let needle = format!("{predicate}(");
    let start = name.find(&needle)? + needle.len();
    let end = name[start..].find(')')? + start;
    Some(name[start..end].to_string())
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    visited: Vec<bool>,
    in_tree: Vec<bool>,
    dist_to_set: Vec<f64>,
}

struct LlmHeuristic {
    places: Vec<String>,
    place_index: HashMap<String, usize>,
    depot_place: usize,
    n_customers: usize,
    customer_place_idx: Vec<usize>,
    demands: Vec<f64>,
    direct: Vec<Vec<f64>>,
    via: Vec<Vec<f64>>,
    to_depot: Vec<f64>,
    loc_fact_by_place: Vec<Option<(usize, usize)>>,
    visited_fact_by_customer: Vec<Option<(usize, usize)>>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    capacity: f64,
    max_vehicles: f64,
    scratch: RefCell<ScratchBuffers>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let operators = task.get_operators();
        let mut place_set = BTreeSet::new();
        let mut customer_set = BTreeSet::new();

        for op in operators.iter() {
            let tokens: Vec<&str> = op.name().split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            if tokens[0] == "visit" && tokens.len() >= 3 {
                place_set.insert(tokens[1].to_string());
                place_set.insert(tokens[2].to_string());
                customer_set.insert(tokens[2].to_string());
            } else if tokens[0] == "visit-via-depot" && tokens.len() >= 3 {
                place_set.insert(tokens[1].to_string());
                place_set.insert(tokens[2].to_string());
                customer_set.insert(tokens[2].to_string());
            } else if tokens[0] == "return-to-depot" && tokens.len() >= 2 {
                place_set.insert(tokens[1].to_string());
            }
        }

        if place_set.is_empty() || customer_set.is_empty() {
            return Err("Could not discover CVRP places/customers".to_string().into());
        }

        let places: Vec<String> = place_set.into_iter().collect();
        let place_index: HashMap<String, usize> = places
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let depot_place = places
            .iter()
            .position(|p| !customer_set.contains(p.as_str()))
            .ok_or_else(|| HeuristicBuildError::from("Could not identify depot place".to_string()))?;
        let customers: Vec<String> = customer_set.into_iter().collect();
        let customer_place_idx: Vec<usize> = customers
            .iter()
            .map(|n| *place_index.get(n.as_str()).unwrap_or(&0))
            .collect();
        let customer_index: HashMap<String, usize> = customers
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let n_customers = customers.len();
        let n_places = places.len();

        let mut direct = vec![vec![INF; n_places]; n_places];
        let mut via = vec![vec![INF; n_places]; n_places];
        let mut to_depot = vec![INF; n_places];

        for op in operators.iter() {
            let tokens: Vec<&str> = op.name().split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            let mut cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                cost = INF;
            }
            if tokens[0] == "visit" && tokens.len() >= 3 {
                if let (Some(&f), Some(&t)) = (
                    place_index.get(tokens[1]),
                    place_index.get(tokens[2]),
                ) {
                    direct[f][t] = cost;
                }
            } else if tokens[0] == "visit-via-depot" && tokens.len() >= 3 {
                if let (Some(&f), Some(&t)) = (
                    place_index.get(tokens[1]),
                    place_index.get(tokens[2]),
                ) {
                    via[f][t] = cost;
                }
            } else if tokens[0] == "return-to-depot" && tokens.len() >= 2 {
                if let Some(&f) = place_index.get(tokens[1]) {
                    to_depot[f] = cost;
                    direct[f][depot_place] = cost;
                }
            }
        }

        for f in 0..n_places {
            if to_depot[f] >= INF * 0.5 && direct[f][depot_place] < INF * 0.5 {
                to_depot[f] = direct[f][depot_place];
            }
        }
        for f in 0..n_places {
            for t in 0..n_places {
                if via[f][t] >= INF * 0.5
                    && direct[f][depot_place] < INF * 0.5
                    && direct[depot_place][t] < INF * 0.5
                {
                    via[f][t] = direct[f][depot_place] + direct[depot_place][t];
                }
            }
        }

        let mut loc_fact_by_place = vec![None; n_places];
        let mut visited_fact_by_customer = vec![None; n_customers];
        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(arg) = extract_arg(&fact_name, "loc") {
                    if let Some(&pi) = place_index.get(&arg) {
                        loc_fact_by_place[pi] = Some((var_id, value));
                    }
                }
                if let Some(arg) = extract_arg(&fact_name, "visited") {
                    if let Some(&ci) = customer_index.get(&arg) {
                        visited_fact_by_customer[ci] = Some((var_id, value));
                    }
                }
            }
        }

        let init_numeric = task.get_initial_numeric_state_values();
        let mut load_var = None;
        let mut vehicles_var = None;
        let mut capacity = INF;
        let mut max_vehicles = INF;

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if name.contains("max_vehicles") {
                if let Some(&v) = init_numeric.get(id) {
                    if v.is_finite() && v >= 0.0 {
                        max_vehicles = v;
                    }
                }
            } else if name.contains("vehicles") {
                if vehicles_var.is_none() {
                    vehicles_var = Some(id);
                }
            } else if name.contains("capacity") {
                if let Some(&v) = init_numeric.get(id) {
                    if v.is_finite() && v >= 0.0 {
                        capacity = v;
                    }
                }
            } else if name.contains("load") {
                if load_var.is_none() {
                    load_var = Some(id);
                }
            }
        }

        let mut demands = vec![0.0; n_customers];
        if let Some(load_id) = load_var {
            for op in operators.iter() {
                let tokens: Vec<&str> = op.name().split_whitespace().collect();
                if tokens.len() >= 3
                    && (tokens[0] == "visit" || tokens[0] == "visit-via-depot")
                {
                    if let Some(&ci) = customer_index.get(tokens[2]) {
                        for assign in op.assignment_effects() {
                            if assign.affected_var_id() == load_id {
                                if let Some(&v) = init_numeric.get(assign.var_id()) {
                                    demands[ci] = v;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if let Some(arg) = extract_arg(&name, "demand") {
                if let Some(&ci) = customer_index.get(&arg) {
                    if demands[ci] == 0.0 {
                        if let Some(&v) = init_numeric.get(id) {
                            demands[ci] = v;
                        }
                    }
                }
            }
        }

        let scratch = RefCell::new(ScratchBuffers {
            propositional: vec![0; task.variables().len()],
            numeric: vec![0.0; task.numeric_variables().len()],
            visited: vec![false; n_customers],
            in_tree: vec![false; n_customers],
            dist_to_set: vec![INF; n_customers],
        });

        Ok(Self {
            places,
            place_index,
            depot_place,
            n_customers,
            customer_place_idx,
            demands,
            direct,
            via,
            to_depot,
            loc_fact_by_place,
            visited_fact_by_customer,
            load_var,
            vehicles_var,
            capacity,
            max_vehicles,
            scratch,
        })
    }

    fn current_place(&self, props: &[usize]) -> usize {
        for (pi, fact) in self.loc_fact_by_place.iter().enumerate() {
            if let Some((vid, val)) = fact {
                if props.get(*vid).copied() == Some(*val) {
                    return pi;
                }
            }
        }
        self.depot_place
    }

    fn return_cost(&self, place: usize) -> f64 {
        if place == self.depot_place {
            return 0.0;
        }
        if self.to_depot[place] < INF * 0.5 {
            self.to_depot[place]
        } else if self.direct[place][self.depot_place] < INF * 0.5 {
            self.direct[place][self.depot_place]
        } else {
            0.0
        }
    }

    fn sym_conn_weight(&self, a: usize, b: usize) -> f64 {
        let mut w = INF;
        w = w.min(self.direct[a][b]);
        w = w.min(self.direct[b][a]);
        w = w.min(self.via[a][b]);
        w = w.min(self.via[b][a]);
        w
    }

    fn min_start_edge(&self, start: usize, visited: &[bool]) -> f64 {
        let mut best = INF;
        if start == self.depot_place {
            for i in 0..self.n_customers {
                if !visited[i] {
                    best = best.min(self.direct[start][self.customer_place_idx[i]]);
                }
            }
        } else {
            for i in 0..self.n_customers {
                if !visited[i] {
                    let p = self.customer_place_idx[i];
                    best = best.min(self.direct[start][p]);
                    best = best.min(self.via[start][p]);
                }
            }
        }
        if best >= INF * 0.5 {
            0.0
        } else {
            best
        }
    }

    fn min_end_return(&self, visited: &[bool]) -> f64 {
        let mut best = INF;
        for i in 0..self.n_customers {
            if !visited[i] {
                best = best.min(self.to_depot[self.customer_place_idx[i]]);
            }
        }
        if best >= INF * 0.5 {
            0.0
        } else {
            best
        }
    }

    fn mst_lower_bound(
        &self,
        visited: &[bool],
        start: usize,
        in_tree: &mut [bool],
        dist: &mut [f64],
    ) -> f64 {
        let remaining = visited.iter().filter(|&&v| !v).count();
        if remaining == 0 {
            return if start == self.depot_place {
                0.0
            } else {
                self.return_cost(start)
            };
        }
        for i in 0..self.n_customers {
            in_tree[i] = false;
            dist[i] = INF;
        }
        let first = match (0..self.n_customers).find(|&i| !visited[i]) {
            Some(i) => i,
            None => return 0.0,
        };
        dist[first] = 0.0;
        let mut total = 0.0;

        for _ in 0..remaining {
            let mut cur = None;
            let mut best = INF;
            for i in 0..self.n_customers {
                if !visited[i] && !in_tree[i] && dist[i] < best {
                    best = dist[i];
                    cur = Some(i);
                }
            }
            let u = match cur {
                Some(u) => u,
                None => break,
            };
            in_tree[u] = true;
            total += best;
            let pu = self.customer_place_idx[u];
            for v in 0..self.n_customers {
                if !visited[v] && !in_tree[v] {
                    let pv = self.customer_place_idx[v];
                    let w = self.sym_conn_weight(pu, pv);
                    if w < dist[v] {
                        dist[v] = w;
                    }
                }
            }
        }

        total + self.min_start_edge(start, visited) + self.min_end_return(visited)
    }

    fn greedy_cost(
        &self,
        start: usize,
        visited: &mut [bool],
        initial_load: f64,
        initial_vehicles: f64,
    ) -> f64 {
        let mut current = start;
        let mut load = initial_load.max(0.0);
        let mut vehicles = initial_vehicles.max(0.0);
        let mut cost = 0.0;
        let mut remaining = visited.iter().filter(|&&v| !v).count();

        while remaining > 0 {
            let mut best_feasible: Option<(f64, usize, bool)> = None;
            let mut best_relaxed: Option<(f64, usize, bool)> = None;

            for i in 0..self.n_customers {
                if visited[i] {
                    continue;
                }
                let p = self.customer_place_idx[i];
                let d = self.direct[current][p];
                let v = if current != self.depot_place {
                    self.via[current][p]
                } else {
                    INF
                };

                let direct_ok = d < INF * 0.5
                    && (self.capacity >= INF * 0.5
                        || load + self.demands[i] <= self.capacity + EPS);
                let via_ok = current != self.depot_place
                    && v < INF * 0.5
                    && (self.max_vehicles >= INF * 0.5
                        || vehicles + 1.0 <= self.max_vehicles + EPS);

                if direct_ok {
                    let better = match best_feasible {
                        Some((bc, _, _)) => d < bc,
                        None => true,
                    };
                    if better {
                        best_feasible = Some((d, i, false));
                    }
                }
                if via_ok {
                    let better = match best_feasible {
                        Some((bc, _, _)) => v < bc,
                        None => true,
                    };
                    if better {
                        best_feasible = Some((v, i, true));
                    }
                }

                let mut relaxed_cost = INF;
                let mut relaxed_via = false;
                if d < INF * 0.5 {
                    relaxed_cost = d;
                }
                if current != self.depot_place && v < INF * 0.5 && v < relaxed_cost {
                    relaxed_cost = v;
                    relaxed_via = true;
                }
                if relaxed_cost < INF * 0.5 {
                    let better = match best_relaxed {
                        Some((bc, _, _)) => relaxed_cost < bc,
                        None => true,
                    };
                    if better {
                        best_relaxed = Some((relaxed_cost, i, relaxed_via));
                    }
                }
            }

            let chosen = match best_feasible.or(best_relaxed) {
                Some(x) => x,
                None => break,
            };
            let (add, cust, use_via) = chosen;
            cost += add;
            if use_via {
                vehicles += 1.0;
                load = self.demands[cust];
            } else {
                load += self.demands[cust];
            }
            current = self.customer_place_idx[cust];
            visited[cust] = true;
            remaining -= 1;
        }

        if remaining == 0 {
            if current != self.depot_place {
                cost += self.return_cost(current);
            }
        } else {
            cost += remaining as f64 * 1000.0;
        }
        cost
    }

    fn sanitize_cost(&self, value: f64) -> f64 {
        if !value.is_finite() || value < 0.0 {
            0.0
        } else if value > 1e12 {
            1e12
        } else {
            value
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

        let mut scratch = self.scratch.borrow_mut();
        registry
            .view(state)
            .fill_numeric(&mut scratch.numeric)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;
        let _ = state.fill_state(registry, &mut scratch.propositional);

        let current_place = self.current_place(&scratch.propositional);
        let mut remaining = 0usize;
        for i in 0..self.n_customers {
            let seen = match self.visited_fact_by_customer[i] {
                Some((vid, val)) => scratch.propositional.get(vid).copied() == Some(val),
                None => false,
            };
            scratch.visited[i] = seen;
            if !seen {
                remaining += 1;
            }
        }

        let current_load = self
            .load_var
            .and_then(|id| scratch.numeric.get(id))
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let vehicles_used = self
            .vehicles_var
            .and_then(|id| scratch.numeric.get(id))
            .copied()
            .unwrap_or(1.0)
            .max(0.0);

        let h = if remaining == 0 {
            if current_place != self.depot_place {
                self.return_cost(current_place)
            } else {
                0.0
            }
        } else {
            let s = &mut *scratch;
            let mst = self.mst_lower_bound(
                &s.visited,
                current_place,
                &mut s.in_tree,
                &mut s.dist_to_set,
            );
            let greedy = self.greedy_cost(
                current_place,
                &mut s.visited,
                current_load,
                vehicles_used,
            );
            0.5 * (mst + greedy)
        };

        Ok(self.sanitize_cost(h))
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
