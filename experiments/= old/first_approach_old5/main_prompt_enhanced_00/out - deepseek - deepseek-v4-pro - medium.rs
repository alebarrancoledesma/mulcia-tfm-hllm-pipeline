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

#[derive(Clone, Copy)]
struct FactRef {
    var_id: usize,
    value: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    remaining: Vec<bool>,
}

struct LlmHeuristic {
    n_customers: usize,
    n_nodes: usize,
    loc_facts: Vec<Option<FactRef>>,
    visited_facts: Vec<FactRef>,
    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    direct_cost: Vec<f64>,
    via_cost: Vec<f64>,
    scratch: RefCell<ScratchBuffers>,
}

fn parse_atom_arg<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = name.strip_prefix(prefix)?;
    rest.strip_suffix(')')
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut visited_map: HashMap<String, (usize, usize)> = HashMap::new();
        let mut loc_map: HashMap<String, (usize, usize)> = HashMap::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(arg) = parse_atom_arg(&fact_name, "Atom visited(") {
                    visited_map.entry(arg.to_string()).or_insert((var_id, value));
                } else if let Some(arg) = parse_atom_arg(&fact_name, "Atom loc(") {
                    loc_map.entry(arg.to_string()).or_insert((var_id, value));
                }
            }
        }

        let mut customer_names: Vec<String> = visited_map.keys().cloned().collect();
        customer_names.sort();
        customer_names.dedup();
        let n_customers = customer_names.len();
        if n_customers == 0 {
            return Err("llm_001: no customer visited facts found".to_string().into());
        }
        let n_nodes = n_customers + 1;

        let depot_name: String = loc_map
            .keys()
            .find(|name| !customer_names.contains(*name))
            .cloned()
            .unwrap_or_else(|| "d1".to_string());

        let mut node_map: HashMap<String, usize> = HashMap::new();
        node_map.insert(depot_name.clone(), 0);
        for (idx, name) in customer_names.iter().enumerate() {
            node_map.insert(name.clone(), idx + 1);
        }

        let mut loc_facts: Vec<Option<FactRef>> = vec![None; n_nodes];
        if let Some(&(var_id, value)) = loc_map.get(&depot_name) {
            loc_facts[0] = Some(FactRef { var_id, value });
        }

        let mut visited_facts: Vec<FactRef> = Vec::with_capacity(n_customers);
        for (idx, name) in customer_names.iter().enumerate() {
            if let Some(&(var_id, value)) = visited_map.get(name) {
                visited_facts.push(FactRef { var_id, value });
            } else {
                return Err(format!("llm_001: missing visited fact for customer {name}").into());
            }

            if let Some(&(var_id, value)) = loc_map.get(name) {
                loc_facts[idx + 1] = Some(FactRef { var_id, value });
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;
        let mut load_var = None;
        let mut vehicles_var = None;
        let mut demands = vec![0.0; n_customers];

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let raw = var.name().to_string();
            let raw = raw.trim();
            let short = raw.strip_prefix("PNE ").unwrap_or(raw);

            if short == "load()" {
                load_var = Some(id);
            } else if short == "vehicles()" {
                vehicles_var = Some(id);
            } else if short == "capacity()" {
                if let Some(&v) = initial_numeric.get(id) {
                    capacity = v;
                }
            } else if short == "max_vehicles()" {
                if let Some(&v) = initial_numeric.get(id) {
                    max_vehicles = v;
                }
            } else if let Some(arg) = parse_atom_arg(short, "demand(") {
                if let Some(&node_idx) = node_map.get(arg) {
                    if node_idx > 0 {
                        demands[node_idx - 1] = initial_numeric.get(id).copied().unwrap_or(0.0);
                    }
                }
            }
        }

        let matrix_size = n_nodes * n_nodes;
        let mut direct_cost = vec![f64::INFINITY; matrix_size];
        let mut via_cost = vec![f64::INFINITY; matrix_size];

        for i in 0..n_nodes {
            direct_cost[i * n_nodes + i] = 0.0;
            via_cost[i * n_nodes + i] = 0.0;
        }

        for op in task.get_operators().iter() {
            let name = op.name().to_string();
            let name = name.trim();
            let tokens: Vec<&str> = name.split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, op).max(0.0);

            if name.starts_with("visit-via-depot") && tokens.len() >= 3 {
                if let (Some(&a), Some(&b)) = (node_map.get(tokens[1]), node_map.get(tokens[2])) {
                    let idx = a * n_nodes + b;
                    if cost.is_finite() && cost < via_cost[idx] {
                        via_cost[idx] = cost;
                    }
                }
            } else if name.starts_with("visit ") && tokens.len() >= 3 {
                if let (Some(&a), Some(&b)) = (node_map.get(tokens[1]), node_map.get(tokens[2])) {
                    let idx = a * n_nodes + b;
                    if cost.is_finite() && cost < direct_cost[idx] {
                        direct_cost[idx] = cost;
                    }
                }
            } else if name.starts_with("return-to-depot") && tokens.len() >= 2 {
                if let Some(&a) = node_map.get(tokens[1]) {
                    let idx = a * n_nodes;
                    if cost.is_finite() && cost < direct_cost[idx] {
                        direct_cost[idx] = cost;
                    }
                }
            }
        }

        for i in 1..n_nodes {
            for j in 1..n_nodes {
                let idx = i * n_nodes + j;
                if !via_cost[idx].is_finite() {
                    let back = direct_cost[i * n_nodes];
                    let out = direct_cost[j];
                    if back.is_finite() && out.is_finite() {
                        via_cost[idx] = back + out;
                    }
                }
            }
        }

        let scratch = ScratchBuffers {
            propositional: vec![0; task.variables().len()],
            numeric: vec![0.0; task.numeric_variables().len()],
            remaining: vec![false; n_nodes],
        };

        Ok(Self {
            n_customers,
            n_nodes,
            loc_facts,
            visited_facts,
            demands,
            capacity,
            max_vehicles,
            load_var,
            vehicles_var,
            direct_cost,
            via_cost,
            scratch: RefCell::new(scratch),
        })
    }

    fn direct(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }
        self.direct_cost
            .get(from * self.n_nodes + to)
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    fn via(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }
        self.via_cost
            .get(from * self.n_nodes + to)
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    fn greedy_route(
        &self,
        start: usize,
        mut load: f64,
        mut vehicles: f64,
        remaining: &mut [bool],
        mut count: usize,
    ) -> f64 {
        let mut cur = start;
        let mut cost = 0.0;

        while count > 0 {
            let mut best_idx = 0usize;
            let mut best_cost = f64::INFINITY;
            let mut best_via = false;
            let mut best_demand = 0.0;

            for i in 0..self.n_customers {
                let node = i + 1;
                if !remaining[node] {
                    continue;
                }

                let demand = self.demands[i];
                let d = self.direct(cur, node);
                let direct_ok = d.is_finite() && load + demand <= self.capacity + 1e-9;
                if direct_ok && d < best_cost {
                    best_cost = d;
                    best_idx = node;
                    best_via = false;
                    best_demand = demand;
                }

                if cur != 0 {
                    let v = self.via(cur, node);
                    let via_ok = v.is_finite()
                        && demand <= self.capacity + 1e-9
                        && vehicles + 1.0 <= self.max_vehicles + 1e-9;
                    if via_ok && v < best_cost {
                        best_cost = v;
                        best_idx = node;
                        best_via = true;
                        best_demand = demand;
                    }
                }
            }

            if best_idx == 0 {
                let mut fallback_cost = f64::INFINITY;
                let mut fallback_idx = 0usize;
                let mut fallback_via = false;
                let mut fallback_demand = 0.0;

                for i in 0..self.n_customers {
                    let node = i + 1;
                    if !remaining[node] {
                        continue;
                    }

                    let demand = self.demands[i];
                    let d = self.direct(cur, node);
                    if d.is_finite() && d < fallback_cost {
                        fallback_cost = d;
                        fallback_idx = node;
                        fallback_via = false;
                        fallback_demand = demand;
                    }

                    if cur != 0 {
                        let v = self.via(cur, node);
                        if v.is_finite() && v < fallback_cost {
                            fallback_cost = v;
                            fallback_idx = node;
                            fallback_via = true;
                            fallback_demand = demand;
                        }
                    }
                }

                if fallback_idx == 0 {
                    break;
                }

                best_idx = fallback_idx;
                best_cost = fallback_cost + 1_000_000_000.0;
                best_via = fallback_via;
                best_demand = fallback_demand;
            }

            cost += best_cost;

            if best_via {
                load = best_demand;
                vehicles += 1.0;
            } else {
                load += best_demand;
            }

            cur = best_idx;
            remaining[best_idx] = false;
            count -= 1;
        }

        if cur != 0 {
            let ret = self.direct(cur, 0);
            if ret.is_finite() {
                cost += ret;
            }
        }

        cost
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

        let mut scratch = self.scratch.borrow_mut();

        state.fill_state(registry, &mut scratch.propositional);

        registry
            .view(state)
            .fill_numeric(&mut scratch.numeric)
            .map_err(|err| EvaluationError::ComputationFailed(format!("{err:?}")))?;

        let mut cur = 0usize;
        for (idx, fact_opt) in self.loc_facts.iter().enumerate() {
            if let Some(fact) = fact_opt {
                let val = scratch.propositional.get(fact.var_id).copied().unwrap_or(usize::MAX);
                if val == fact.value {
                    cur = idx;
                    break;
                }
            }
        }

        let remaining = &mut scratch.remaining;
        for i in 0..self.n_customers {
            let fact = &self.visited_facts[i];
            let val = scratch.propositional.get(fact.var_id).copied().unwrap_or(usize::MAX);
            remaining[i + 1] = val != fact.value;
        }

        let mut count = 0usize;
        for i in 1..self.n_nodes {
            if remaining[i] {
                count += 1;
            }
        }

        let mut load = 0.0;
        if let Some(id) = self.load_var {
            let v = scratch.numeric.get(id).copied().unwrap_or(0.0);
            if v.is_finite() && v >= 0.0 {
                load = v;
            }
        }

        let mut vehicles = 1.0;
        if let Some(id) = self.vehicles_var {
            let v = scratch.numeric.get(id).copied().unwrap_or(1.0);
            if v.is_finite() && v >= 1.0 {
                vehicles = v;
            }
        }

        let mut h = self.greedy_route(cur, load, vehicles, remaining, count);
        if !h.is_finite() || h < 0.0 {
            h = 0.0;
        }

        Ok(h)
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
