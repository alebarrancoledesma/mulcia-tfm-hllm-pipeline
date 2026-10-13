use std::cell::RefCell;
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

const INF: f64 = 1.0e18;

fn parse_atom_arg(s: &str, predicate: &str) -> Option<String> {
    let t = s.trim();
    let needle = format!("Atom {}(", predicate);
    if t.starts_with(&needle) && t.ends_with(')') {
        let inner = &t[needle.len()..t.len()-1];
        if !inner.is_empty() {
            return Some(inner.trim().to_string());
        }
    }
    None
}

fn name_eq(raw: &str, short: &str) -> bool {
    let t = raw.trim();
    if t == short {
        return true;
    }
    if let Some(rest) = t.strip_prefix("PNE ") {
        return rest == short;
    }
    false
}

fn parse_demand_name(raw: &str) -> Option<String> {
    let t = raw.trim();
    for prefix in ["demand(", "PNE demand("] {
        if t.starts_with(prefix) && t.ends_with(')') {
            let inner = &t[prefix.len()..t.len()-1];
            if !inner.is_empty() {
                return Some(inner.trim().to_string());
            }
        }
    }
    None
}

struct CustomerInfo {
    name: String,
    demand: f64,
    visited_var_id: usize,
    visited_value: usize,
}

struct ScratchBuffers {
    prop: Vec<usize>,
    numeric: Vec<f64>,
    remaining: Vec<usize>,
}

struct LlmHeuristic {
    n_customers: usize,
    node_count: usize,
    depot_idx: usize,
    customers: Vec<CustomerInfo>,
    loc_var_id: usize,
    loc_value_to_node: Vec<Option<usize>>,
    direct: Vec<f64>,
    via: Vec<f64>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    capacity: f64,
    max_vehicles: f64,
    prop_var_count: usize,
    numeric_var_count: usize,
    scratch: RefCell<ScratchBuffers>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let variables = task.variables();
        let prop_var_count = variables.len();
        let mut visited_fact_by_customer: HashMap<String, (usize, usize)> = HashMap::new();
        let mut loc_fact_by_place: HashMap<String, (usize, usize)> = HashMap::new();
        let mut loc_var_id: Option<usize> = None;
        let mut max_domain_size = 0usize;

        for var_id in 0..variables.len() {
            let var = &variables[var_id];
            let ds = var.domain_size();
            if ds > max_domain_size {
                max_domain_size = ds;
            }
            for value in 0..ds {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(inner) = parse_atom_arg(&fact_name, "visited") {
                    visited_fact_by_customer.entry(inner).or_insert((var_id, value));
                } else if let Some(inner) = parse_atom_arg(&fact_name, "loc") {
                    loc_fact_by_place.entry(inner).or_insert((var_id, value));
                    if loc_var_id.is_none() {
                        loc_var_id = Some(var_id);
                    }
                }
            }
        }

        let mut customer_names: Vec<String> = visited_fact_by_customer.keys().cloned().collect();
        customer_names.sort();
        let n_customers = customer_names.len();
        let mut customers = Vec::with_capacity(n_customers);
        for name in customer_names {
            let (vid, vval) = visited_fact_by_customer[&name];
            customers.push(CustomerInfo {
                name: name.clone(),
                demand: 0.0,
                visited_var_id: vid,
                visited_value: vval,
            });
        }

        let mut depot_name: Option<String> = None;
        for place in loc_fact_by_place.keys() {
            if !visited_fact_by_customer.contains_key(place) {
                depot_name = Some(place.clone());
                break;
            }
        }
        let depot_name = match depot_name {
            Some(name) => name,
            None => return Err(HeuristicBuildError::from("missing depot loc fact".to_string())),
        };

        let loc_var_id = match loc_var_id {
            Some(id) => id,
            None => return Err(HeuristicBuildError::from("missing loc variable".to_string())),
        };

        let depot_idx = n_customers;
        let node_count = n_customers + 1;
        let mut name_to_node: HashMap<String, usize> = HashMap::new();
        for (i, customer) in customers.iter().enumerate() {
            name_to_node.insert(customer.name.clone(), i);
        }
        name_to_node.insert(depot_name.clone(), depot_idx);

        let mut loc_value_to_node: Vec<Option<usize>> = vec![None; max_domain_size];
        for (place, (_, value)) in loc_fact_by_place.iter() {
            if let Some(&node) = name_to_node.get(place) {
                if *value < loc_value_to_node.len() {
                    loc_value_to_node[*value] = Some(node);
                }
            }
        }

        let numeric_variables = task.numeric_variables();
        let numeric_var_count = numeric_variables.len();
        let initial_numeric_values = task.get_initial_numeric_state_values();
        let mut load_var: Option<usize> = None;
        let mut vehicles_var: Option<usize> = None;
        let mut capacity: f64 = f64::MAX;
        let mut max_vehicles: f64 = f64::MAX;
        let mut exact_demand_by_name: HashMap<String, f64> = HashMap::new();

        for (id, var) in numeric_variables.iter().enumerate() {
            let raw = var.name();
            if name_eq(raw, "load()") {
                if load_var.is_none() {
                    load_var = Some(id);
                }
            }
            if name_eq(raw, "vehicles()") {
                if vehicles_var.is_none() {
                    vehicles_var = Some(id);
                }
            }
            if name_eq(raw, "capacity()") {
                capacity = initial_numeric_values.get(id).copied().unwrap_or(f64::MAX);
            }
            if name_eq(raw, "max_vehicles()") {
                max_vehicles = initial_numeric_values.get(id).copied().unwrap_or(f64::MAX);
            }
            if let Some(cname) = parse_demand_name(raw) {
                exact_demand_by_name.insert(cname, initial_numeric_values.get(id).copied().unwrap_or(0.0));
            }
        }

        let mut demand_known = vec![false; n_customers];
        for (i, customer) in customers.iter_mut().enumerate() {
            if let Some(&d) = exact_demand_by_name.get(&customer.name) {
                customer.demand = d;
                demand_known[i] = true;
            }
        }

        if let Some(load_id) = load_var {
            let ops = task.get_operators();
            for op in ops {
                let op_name = op.name();
                let tokens: Vec<&str> = op_name.split_whitespace().collect();
                if tokens.len() != 3 {
                    continue;
                }
                if tokens[0] != "visit" && tokens[0] != "visit-via-depot" {
                    continue;
                }
                let to_name = tokens[2];
                let idx = match name_to_node.get(to_name) {
                    Some(&i) if i < n_customers => i,
                    _ => continue,
                };
                if demand_known[idx] {
                    continue;
                }
                for assign in op.assignment_effects() {
                    if assign.affected_var_id() == load_id {
                        let rhs = assign.var_id();
                        if rhs < initial_numeric_values.len() {
                            customers[idx].demand = initial_numeric_values[rhs];
                            demand_known[idx] = true;
                            break;
                        }
                    }
                }
            }
        }

        let mut direct = vec![INF; node_count * node_count];
        let mut via = vec![INF; node_count * node_count];
        for i in 0..node_count {
            direct[i * node_count + i] = 0.0;
        }

        let ops = task.get_operators();
        for op in ops {
            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }
            let op_name = op.name();
            let tokens: Vec<&str> = op_name.split_whitespace().collect();
            if tokens.len() == 3 && tokens[0] == "visit" {
                let from = name_to_node.get(tokens[1]).copied();
                let to = name_to_node.get(tokens[2]).copied();
                if let (Some(f), Some(t)) = (from, to) {
                    let idx = f * node_count + t;
                    if cost < direct[idx] {
                        direct[idx] = cost;
                    }
                }
            } else if tokens.len() == 2 && tokens[0] == "return-to-depot" {
                let from = name_to_node.get(tokens[1]).copied();
                if let Some(f) = from {
                    let idx = f * node_count + depot_idx;
                    if cost < direct[idx] {
                        direct[idx] = cost;
                    }
                }
            } else if tokens.len() == 3 && tokens[0] == "visit-via-depot" {
                let from = name_to_node.get(tokens[1]).copied();
                let to = name_to_node.get(tokens[2]).copied();
                if let (Some(f), Some(t)) = (from, to) {
                    let idx = f * node_count + t;
                    if cost < via[idx] {
                        via[idx] = cost;
                    }
                }
            }
        }

        for f in 0..n_customers {
            for t in 0..n_customers {
                if f == t {
                    continue;
                }
                let idx = f * node_count + t;
                if via[idx] >= INF * 0.5 {
                    let d1 = direct[f * node_count + depot_idx];
                    let d2 = direct[depot_idx * node_count + t];
                    if d1 < INF * 0.5 && d2 < INF * 0.5 {
                        via[idx] = d1 + d2;
                    }
                }
            }
        }

        Ok(Self {
            n_customers,
            node_count,
            depot_idx,
            customers,
            loc_var_id,
            loc_value_to_node,
            direct,
            via,
            load_var,
            vehicles_var,
            capacity,
            max_vehicles,
            prop_var_count,
            numeric_var_count,
            scratch: RefCell::new(ScratchBuffers {
                prop: Vec::new(),
                numeric: Vec::new(),
                remaining: Vec::new(),
            }),
        })
    }

    #[inline]
    fn direct_cost(&self, from: usize, to: usize) -> f64 {
        self.direct[from * self.node_count + to]
    }

    #[inline]
    fn via_cost(&self, from: usize, to: usize) -> f64 {
        self.via[from * self.node_count + to]
    }

    fn greedy_estimate(
        &self,
        start: usize,
        start_load: f64,
        start_vehicles: f64,
        remaining: &mut Vec<usize>,
    ) -> f64 {
        let inf = INF * 0.5;
        let eps = 1.0e-9;
        let mut cur = start;
        let mut load = start_load.max(0.0);
        let mut vehicles = start_vehicles.max(0.0);
        let mut cost = 0.0_f64;

        while !remaining.is_empty() {
            let at_depot = cur == self.depot_idx;
            let mut best_pos: Option<usize> = None;
            let mut best_cost = INF;
            let mut best_via = false;

            for pos in 0..remaining.len() {
                let c = remaining[pos];
                let dem = self.customers[c].demand;

                if at_depot {
                    if dem <= self.capacity - load + eps {
                        let d = self.direct_cost(self.depot_idx, c);
                        if d < inf && d < best_cost {
                            best_cost = d;
                            best_pos = Some(pos);
                            best_via = false;
                        }
                    }
                } else {
                    if dem <= self.capacity - load + eps {
                        let d = self.direct_cost(cur, c);
                        if d < inf && d < best_cost {
                            best_cost = d;
                            best_pos = Some(pos);
                            best_via = false;
                        }
                    }
                    if vehicles + 1.0 <= self.max_vehicles + eps {
                        let d = self.via_cost(cur, c);
                        if d < inf && d < best_cost {
                            best_cost = d;
                            best_pos = Some(pos);
                            best_via = true;
                        }
                    }
                }
            }

            if best_pos.is_none() {
                best_cost = INF;
                for pos in 0..remaining.len() {
                    let c = remaining[pos];
                    let ddir = if at_depot {
                        self.direct_cost(self.depot_idx, c)
                    } else {
                        self.direct_cost(cur, c)
                    };
                    let dvia = if at_depot {
                        INF
                    } else {
                        self.via_cost(cur, c)
                    };
                    let d = if ddir <= dvia { ddir } else { dvia };
                    if d < inf && d < best_cost {
                        best_cost = d;
                        best_pos = Some(pos);
                        best_via = !at_depot && dvia < ddir;
                    }
                }
            }

            let pos = best_pos.unwrap_or(0);
            let c = remaining.remove(pos);
            let dem = self.customers[c].demand;

            if at_depot {
                let d = self.direct_cost(self.depot_idx, c);
                cost += if d < inf { d } else { 0.0 };
                load += dem;
            } else if best_via {
                let d = self.via_cost(cur, c);
                cost += if d < inf { d } else { 0.0 };
                vehicles += 1.0;
                load = dem;
            } else {
                let d = self.direct_cost(cur, c);
                cost += if d < inf { d } else { 0.0 };
                load += dem;
            }
            cur = c;
        }

        if cur != self.depot_idx {
            let d = self.direct_cost(cur, self.depot_idx);
            if d < inf {
                cost += d;
            }
        }

        if !cost.is_finite() || cost < 0.0 {
            0.0
        } else {
            cost
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
        let prop = &mut scratch.prop;
        prop.clear();
        prop.resize(self.prop_var_count, 0);
        state.fill_state(registry, prop);

        let numeric = &mut scratch.numeric;
        numeric.clear();
        numeric.resize(self.numeric_var_count, 0.0);
        registry.view(state).fill_numeric(numeric).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let current = prop
            .get(self.loc_var_id)
            .copied()
            .and_then(|value| self.loc_value_to_node.get(value).copied().flatten())
            .unwrap_or(self.depot_idx);

        let load = self
            .load_var
            .and_then(|id| numeric.get(id).copied())
            .unwrap_or(0.0)
            .max(0.0);
        let vehicles = self
            .vehicles_var
            .and_then(|id| numeric.get(id).copied())
            .unwrap_or(1.0)
            .max(0.0);

        let remaining = &mut scratch.remaining;
        remaining.clear();
        for i in 0..self.n_customers {
            let customer = &self.customers[i];
            let value = prop.get(customer.visited_var_id).copied().unwrap_or(usize::MAX);
            if value != customer.visited_value {
                remaining.push(i);
            }
        }

        if remaining.is_empty() {
            if current == self.depot_idx {
                return Ok(0.0);
            }
            let d = self.direct_cost(current, self.depot_idx);
            return Ok(if d < INF * 0.5 && d >= 0.0 { d } else { 0.0 });
        }

        let h = self.greedy_estimate(current, load, vehicles, remaining);
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
