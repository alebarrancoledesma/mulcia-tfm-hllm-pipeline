use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::task::Fact;

use planforge_search::config::HeuristicSpec;
use planforge_search::evaluation::{
    EvaluationError,
    EvaluationState,
    Heuristic,
    metric_operator_cost_from_initial_values,
};
use planforge_search::heuristic_factory::{
    ExternalHeuristic,
    HeuristicBuildError,
    RequiredBackend,
    TaskRequirements,
};

const LARGE_PENALTY: f64 = 1e12;
const EPS: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpKind { Visit, ViaDepot, Return }

struct ParsedOp {
    kind: OpKind,
    from: String,
    to: Option<String>,
}

fn parse_op_name(name: &str) -> Option<ParsedOp> {
    let parts: Vec<&str> = name.split_whitespace().collect();
    if parts.len() == 3 {
        if parts[0] == "visit" {
            return Some(ParsedOp {
                kind: OpKind::Visit,
                from: parts[1].to_string(),
                to: Some(parts[2].to_string()),
            });
        }
        if parts[0] == "visit-via-depot" {
            return Some(ParsedOp {
                kind: OpKind::ViaDepot,
                from: parts[1].to_string(),
                to: Some(parts[2].to_string()),
            });
        }
    }
    if parts.len() == 2 && parts[0] == "return-to-depot" {
        return Some(ParsedOp {
            kind: OpKind::Return,
            from: parts[1].to_string(),
            to: None,
        });
    }
    None
}

fn strip_wrapped<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = s.strip_prefix(prefix)?;
    rest.strip_suffix(')')
}

fn parse_loc_fact_name(name: &str) -> Option<&str> {
    strip_wrapped(name, "Atom loc(").or_else(|| strip_wrapped(name, "NegatedAtom loc("))
}

fn parse_visited_fact_name(name: &str) -> Option<&str> {
    strip_wrapped(name, "Atom visited(").or_else(|| strip_wrapped(name, "NegatedAtom visited("))
}

fn parse_demand_var_name(name: &str) -> Option<&str> {
    strip_wrapped(name, "demand(")
}

fn parse_travel_cost_var_name(name: &str) -> Option<(String, String)> {
    let rest = strip_wrapped(name, "travel-cost(")?;
    let mut pieces = rest.split(',');
    let a = pieces.next()?.trim().to_string();
    let b = pieces.next()?.trim().to_string();
    if pieces.next().is_some() {
        return None;
    }
    Some((a, b))
}

struct LlmHeuristic {
    loc_facts: Vec<Fact>,
    visited_facts: Vec<Fact>,
    num_customers: usize,
    depot_idx: usize,

    load_id: usize,
    vehicles_id: usize,
    numeric_len: usize,

    capacity: f64,
    max_vehicles: f64,
    has_capacity: bool,
    has_max_vehicles: bool,
    has_demands: bool,
    has_resources: bool,
    demands: Vec<f64>,

    direct_cost: Vec<Vec<f64>>,
    via_cost: Vec<Vec<f64>>,
    return_cost: Vec<f64>,

    direct_sorted: Vec<Vec<usize>>,
    via_sorted: Vec<Vec<usize>>,

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_bool: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut place_set = HashSet::new();
        let mut customer_set = HashSet::new();
        let mut loc_fact_by_place: HashMap<String, Fact> = HashMap::new();
        let mut visited_fact_by_customer: HashMap<String, Fact> = HashMap::new();

        for op in task.get_operators() {
            let op_name = op.name().to_string();
            if let Some(parsed) = parse_op_name(&op_name) {
                place_set.insert(parsed.from.clone());
                if parsed.kind != OpKind::Return {
                    if let Some(to) = &parsed.to {
                        place_set.insert(to.clone());
                        customer_set.insert(to.clone());
                    }
                }
            }

            for fact in op.preconditions().iter().chain(op.effects().iter()) {
                let fname = task.get_fact_name(fact.clone());

                if let Some(place) = parse_loc_fact_name(&fname) {
                    place_set.insert(place.to_string());
                    if fname.starts_with("Atom loc(") {
                        loc_fact_by_place.insert(place.to_string(), fact.clone());
                    }
                }

                if let Some(cust) = parse_visited_fact_name(&fname) {
                    customer_set.insert(cust.to_string());
                    if fname.starts_with("Atom visited(") {
                        visited_fact_by_customer.insert(cust.to_string(), fact.clone());
                    }
                }
            }
        }

        let mut customer_names: Vec<String> = customer_set.into_iter().collect();
        customer_names.sort();

        let non_customers: Vec<String> = place_set
            .iter()
            .filter(|p| !customer_names.iter().any(|c| c == *p))
            .cloned()
            .collect();

        if non_customers.is_empty() {
            return Err("CVRP heuristic could not identify the depot"
                .to_string()
                .into());
        }

        let depot_name = non_customers[0].clone();
        let num_customers = customer_names.len();
        let depot_idx = num_customers;

        let mut place_index: HashMap<String, usize> = HashMap::new();
        for (i, name) in customer_names.iter().enumerate() {
            place_index.insert(name.clone(), i);
        }
        place_index.insert(depot_name.clone(), depot_idx);

        let mut loc_facts = Vec::with_capacity(num_customers + 1);
        for name in &customer_names {
            let f = loc_fact_by_place
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    HeuristicBuildError::from(format!("Missing loc fact for {}", name))
                })?;
            loc_facts.push(f);
        }
        loc_facts.push(
            loc_fact_by_place
                .get(&depot_name)
                .cloned()
                .ok_or_else(|| HeuristicBuildError::from("Missing loc fact for depot"))?,
        );

        let visited_facts: Vec<Fact> = customer_names
            .iter()
            .map(|name| {
                visited_fact_by_customer
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        HeuristicBuildError::from(format!("Missing visited fact for {}", name))
                    })
            })
            .collect::<Result<_, _>>()?;

        let numeric_vars: Vec<_> = task.numeric_variables().collect();
        let init_values = task.get_initial_numeric_state_values();

        let mut load_id = None;
        let mut vehicles_id = None;
        let mut max_vehicles_id = None;
        let mut capacity_id = None;

        let mut demands = vec![0.0; num_customers];
        let mut has_demands = false;

        for nv in &numeric_vars {
            let name = nv.name().to_string();
            let id = nv.id();

            if name == "load()" {
                load_id = Some(id);
            } else if name == "vehicles()" {
                vehicles_id = Some(id);
            } else if name == "max_vehicles()" {
                max_vehicles_id = Some(id);
            } else if name == "capacity()" {
                capacity_id = Some(id);
            }

            if let Some(obj) = parse_demand_var_name(&name) {
                if let Some(&idx) = place_index.get(obj) {
                    if idx < num_customers {
                        demands[idx] = init_values.get(id).copied().unwrap_or(0.0);
                        has_demands = true;
                    }
                }
            }
        }

        let load_id = load_id.ok_or_else(|| {
            HeuristicBuildError::from("Missing numeric variable load()")
        })?;
        let vehicles_id = vehicles_id.ok_or_else(|| {
            HeuristicBuildError::from("Missing numeric variable vehicles()")
        })?;

        let capacity = capacity_id
            .and_then(|id| init_values.get(id).copied())
            .unwrap_or(f64::MAX);
        let max_vehicles = max_vehicles_id
            .and_then(|id| init_values.get(id).copied())
            .unwrap_or(f64::MAX);
        let has_capacity = capacity_id.is_some();
        let has_max_vehicles = max_vehicles_id.is_some();
        let has_resources = has_capacity && has_max_vehicles && has_demands;

        let num_places = num_customers + 1;
        let mut direct_cost = vec![vec![f64::INFINITY; num_customers]; num_places];
        let mut via_cost = vec![vec![f64::INFINITY; num_customers]; num_customers];
        let mut return_cost = vec![f64::INFINITY; num_places];
        return_cost[depot_idx] = 0.0;

        for op in task.get_operators() {
            let op_name = op.name().to_string();
            let Some(parsed) = parse_op_name(&op_name) else {
                continue;
            };
            let cost = metric_operator_cost_from_initial_values(task, op);

            match parsed.kind {
                OpKind::Visit => {
                    let from_idx = *place_index.get(&parsed.from).unwrap();
                    let to_idx = *place_index.get(parsed.to.as_ref().unwrap()).unwrap();
                    direct_cost[from_idx][to_idx] = cost;
                }
                OpKind::ViaDepot => {
                    let from_idx = *place_index.get(&parsed.from).unwrap();
                    let to_idx = *place_index.get(parsed.to.as_ref().unwrap()).unwrap();
                    via_cost[from_idx][to_idx] = cost;
                }
                OpKind::Return => {
                    let from_idx = *place_index.get(&parsed.from).unwrap();
                    return_cost[from_idx] = cost;
                }
            }
        }

        for nv in &numeric_vars {
            let name = nv.name().to_string();
            let Some((from, to)) = parse_travel_cost_var_name(&name) else {
                continue;
            };
            let Some(&from_idx) = place_index.get(&from) else {
                continue;
            };
            let Some(&to_idx) = place_index.get(&to) else {
                continue;
            };
            let value = init_values.get(nv.id()).copied().unwrap_or(0.0);

            if to_idx == depot_idx && from_idx != depot_idx {
                if return_cost[from_idx].is_infinite() {
                    return_cost[from_idx] = value;
                }
            } else if from_idx == depot_idx && to_idx < num_customers {
                if direct_cost[from_idx][to_idx].is_infinite() {
                    direct_cost[from_idx][to_idx] = value;
                }
            } else if from_idx < num_customers && to_idx < num_customers {
                if direct_cost[from_idx][to_idx].is_infinite() {
                    direct_cost[from_idx][to_idx] = value;
                }
            }
        }

        for from in 0..num_places {
            if from == depot_idx {
                continue;
            }
            for to in 0..num_customers {
                if direct_cost[from][to].is_infinite() {
                    let ret = return_cost[from];
                    let out = direct_cost[depot_idx][to];
                    if ret.is_finite() && out.is_finite() {
                        direct_cost[from][to] = ret + out;
                    }
                }
            }
        }

        for from in 0..num_customers {
            for to in 0..num_customers {
                if via_cost[from][to].is_infinite() {
                    let ret = return_cost[from];
                    let out = direct_cost[depot_idx][to];
                    if ret.is_finite() && out.is_finite() {
                        via_cost[from][to] = ret + out;
                    }
                }
            }
        }

        let mut direct_sorted = Vec::with_capacity(num_places);
        for p in 0..num_places {
            let mut list: Vec<usize> = (0..num_customers).collect();
            list.sort_by(|&a, &b| {
                direct_cost[p][a]
                    .partial_cmp(&direct_cost[p][b])
                    .unwrap_or(Ordering::Equal)
            });
            direct_sorted.push(list);
        }

        let mut via_sorted = Vec::with_capacity(num_customers);
        for c in 0..num_customers {
            let mut list: Vec<usize> = (0..num_customers).collect();
            list.sort_by(|&a, &b| {
                via_cost[c][a]
                    .partial_cmp(&via_cost[c][b])
                    .unwrap_or(Ordering::Equal)
            });
            via_sorted.push(list);
        }

        Ok(Self {
            loc_facts,
            visited_facts,
            num_customers,
            depot_idx,
            load_id,
            vehicles_id,
            numeric_len: numeric_vars.len(),
            capacity,
            max_vehicles,
            has_capacity,
            has_max_vehicles,
            has_demands,
            has_resources,
            demands,
            direct_cost,
            via_cost,
            return_cost,
            direct_sorted,
            via_sorted,
            scratch_numeric: RefCell::new(vec![0.0; numeric_vars.len()]),
            scratch_bool: RefCell::new(Vec::new()),
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
            let mut nums = self.scratch_numeric.borrow_mut();
            nums.clear();
            nums.resize(self.numeric_len, 0.0);
            registry.view(state).fill_numeric(&mut nums);
            (nums[self.load_id], nums[self.vehicles_id])
        };

        let mut flags = self.scratch_bool.borrow_mut();
        flags.clear();
        flags.resize(self.num_customers, true);

        let mut current_place = self.depot_idx;
        for (idx, fact) in self.loc_facts.iter().enumerate() {
            if fact.is_hold(registry.view(state)) {
                current_place = idx;
                break;
            }
        }

        let mut remaining_demand = 0.0;
        for c in 0..self.num_customers {
            if self.visited_facts[c].is_hold(registry.view(state)) {
                flags[c] = false;
            } else if self.has_demands {
                remaining_demand += self.demands[c];
            }
        }

        let mut h_capacity = 0.0;
        if self.has_resources {
            let remaining_current_capacity = (self.capacity - load).max(0.0);
            let extra_vehicles = (self.max_vehicles - vehicles).max(0.0);
            let available_capacity = remaining_current_capacity + extra_vehicles * self.capacity;
            if remaining_demand > available_capacity + EPS {
                return Ok(LARGE_PENALTY);
            }
        }

        let mut cur_place = current_place;
        let mut cur_load = load;
        let mut cur_vehicles = vehicles;
        let mut remaining_count = flags.iter().filter(|&&v| v).count();
        let mut h_routing = 0.0;

        while remaining_count > 0 {
            let mut best_cost = f64::INFINITY;
            let mut best_customer = usize::MAX;
            let mut best_is_via = false;

            if cur_place < self.direct_sorted.len() {
                for &cand in &self.direct_sorted[cur_place] {
                    if !flags[cand] {
                        continue;
                    }

                    let fits_directly = !self.has_capacity
                        || self.demands[cand] <= (self.capacity - cur_load) + EPS;
                    if fits_directly {
                        let c = self.direct_cost[cur_place][cand];
                        if c.is_finite() && c < best_cost {
                            best_cost = c;
                            best_customer = cand;
                            best_is_via = false;
                        }
                        break;
                    }
                }
            }

            if cur_place != self.depot_idx
                && (!self.has_max_vehicles
                    || cur_vehicles + 1.0 <= self.max_vehicles + EPS)
            {
                for &cand in &self.via_sorted[cur_place] {
                    if !flags[cand] {
                        continue;
                    }

                    let c = self.via_cost[cur_place][cand];
                    if c.is_finite() && c < best_cost {
                        best_cost = c;
                        best_customer = cand;
                        best_is_via = true;
                    }
                    break;
                }
            }

            if best_customer == usize::MAX || !best_cost.is_finite() {
                return Ok(LARGE_PENALTY);
            }

            h_routing += best_cost;
            if best_is_via {
                cur_load = self.demands[best_customer];
                cur_vehicles += 1.0;
            } else {
                cur_load += self.demands[best_customer];
            }
            cur_place = best_customer;
            flags[best_customer] = false;
            remaining_count -= 1;

            if h_routing > LARGE_PENALTY {
                return Ok(LARGE_PENALTY);
            }
        }

        if cur_place != self.depot_idx {
            let ret = self.return_cost[cur_place];
            if ret.is_finite() {
                h_routing += ret;
            } else {
                return Ok(LARGE_PENALTY);
            }
        }

        Ok((h_routing + h_capacity).max(0.0))
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
