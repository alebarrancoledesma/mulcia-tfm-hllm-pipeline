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
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy)]
struct PropFactRef {
    var_id: usize,
    positive_value: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    unvisited: Vec<usize>,
}

fn parse_fact_name(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix("Atom ")?;
    let rest = rest.trim();
    let open = rest.find('(')?;
    if !rest.ends_with(')') {
        return None;
    }
    let end = rest.len() - 1;
    if open + 1 >= end {
        return None;
    }
    let predicate = &rest[..open];
    let object = &rest[open + 1..end];
    if object.is_empty() || object.contains('(') {
        return None;
    }
    Some((predicate, object))
}

fn parse_operator_name(name: &str) -> Option<(&str, &str, &str)> {
    let parts: Vec<&str> = name.split_whitespace().collect();
    if parts.len() == 3 && parts[0] == "visit" {
        Some(("visit", parts[1], parts[2]))
    } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
        Some(("via", parts[1], parts[2]))
    } else if parts.len() == 2 && parts[0] == "return-to-depot" {
        Some(("return", parts[1], ""))
    } else {
        None
    }
}

struct LlmHeuristic {
    depot_id: usize,
    customer_ids: Vec<usize>,
    visited_facts: Vec<PropFactRef>,
    loc_facts: Vec<Option<PropFactRef>>,
    direct: Vec<Vec<f64>>,
    via: Vec<Vec<f64>>,
    demand_by_node: Vec<f64>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    avg_depot_round_trip: f64,
    scratch: RefCell<ScratchBuffers>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut visited_facts_map: BTreeMap<String, PropFactRef> = BTreeMap::new();
        let mut loc_facts_map: BTreeMap<String, PropFactRef> = BTreeMap::new();
        let mut customers: BTreeSet<String> = BTreeSet::new();

        let variables = task.variables();
        for var_id in 0..variables.len() {
            let domain = variables[var_id].domain_size();
            for value in 0..domain {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some((predicate, object)) = parse_fact_name(&fact_name) {
                    match predicate {
                        "visited" => {
                            visited_facts_map
                                .entry(object.to_string())
                                .or_insert(PropFactRef {
                                    var_id,
                                    positive_value: value,
                                });
                            customers.insert(object.to_string());
                        }
                        "loc" => {
                            loc_facts_map
                                .entry(object.to_string())
                                .or_insert(PropFactRef {
                                    var_id,
                                    positive_value: value,
                                });
                        }
                        _ => {}
                    }
                }
            }
        }

        if customers.is_empty() {
            return Err("No visited-customer facts found".to_string().into());
        }

        let mut depot_name: Option<String> = None;
        for name in loc_facts_map.keys() {
            if !customers.contains(name) {
                depot_name = Some(name.clone());
                break;
            }
        }
        let depot_name = depot_name.ok_or_else(|| {
            "Unable to identify depot".to_string().into()
        })?;

        let customer_names: Vec<String> = customers.into_iter().collect();

        let mut node_id_by_name: HashMap<String, usize> = HashMap::new();
        node_id_by_name.insert(depot_name.clone(), 0);

        let customer_ids: Vec<usize> = customer_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let node_id = idx + 1;
                node_id_by_name.insert(name.clone(), node_id);
                node_id
            })
            .collect();

        let n_nodes = customer_names.len() + 1;

        let mut visited_facts = Vec::with_capacity(customer_ids.len());
        let mut loc_facts: Vec<Option<PropFactRef>> = vec![None; n_nodes];
        loc_facts[0] = loc_facts_map.get(&depot_name).copied();

        for (ci, name) in customer_names.iter().enumerate() {
            let visited_ref = *visited_facts_map
                .get(name)
                .ok_or_else(|| format!("Missing visited fact for {name}").into())?;
            visited_facts.push(visited_ref);
            loc_facts[ci + 1] = loc_facts_map.get(name).copied();
        }

        let mut direct = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
        let mut via = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
        for i in 0..n_nodes {
            direct[i][i] = 0.0;
            via[i][i] = 0.0;
        }

        for op in task.get_operators() {
            let cost = metric_operator_cost_from_initial_values(task, op) as f64;
            if let Some(parsed) = parse_operator_name(op.name()) {
                match parsed.0 {
                    "visit" => {
                        if let (Some(&from), Some(&to)) = (
                            node_id_by_name.get(parsed.1),
                            node_id_by_name.get(parsed.2),
                        ) {
                            if cost < direct[from][to] {
                                direct[from][to] = cost;
                            }
                        }
                    }
                    "via" => {
                        if let (Some(&from), Some(&to)) = (
                            node_id_by_name.get(parsed.1),
                            node_id_by_name.get(parsed.2),
                        ) {
                            if cost < via[from][to] {
                                via[from][to] = cost;
                            }
                        }
                    }
                    "return" => {
                        if let Some(&from) = node_id_by_name.get(parsed.1) {
                            if cost < direct[from][0] {
                                direct[from][0] = cost;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i == j {
                    continue;
                }
                if direct[i][j].is_infinite() {
                    let detour = direct[i][0] + direct[0][j];
                    if detour.is_finite() {
                        direct[i][j] = detour;
                    }
                }
            }
        }
        for i in 1..n_nodes {
            for j in 1..n_nodes {
                if via[i][j].is_infinite() {
                    let detour = direct[i][0] + direct[0][j];
                    if detour.is_finite() {
                        via[i][j] = detour;
                    }
                }
            }
        }

        let large = 1e12;
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if direct[i][j].is_infinite() {
                    direct[i][j] = large;
                }
                if via[i][j].is_infinite() {
                    via[i][j] = large;
                }
            }
        }

        let mut sum_depot_round_trip = 0.0;
        let mut depot_round_trip_count = 0usize;
        for i in 1..n_nodes {
            let round_trip = direct[i][0] + direct[0][i];
            if round_trip.is_finite() {
                sum_depot_round_trip += round_trip;
                depot_round_trip_count += 1;
            }
        }
        let avg_depot_round_trip = if depot_round_trip_count > 0 {
            sum_depot_round_trip / depot_round_trip_count as f64
        } else {
            large
        };

        let mut load_var: Option<usize> = None;
        let mut vehicles_var: Option<usize> = None;
        let mut capacity_var: Option<usize> = None;
        let mut max_vehicles_var: Option<usize> = None;

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name();
            if name.contains("max_vehicles")
                || name.contains("max-vehicles")
                || name.contains("max_vehicle")
            {
                max_vehicles_var = Some(id);
            } else if name.contains("vehicles") {
                vehicles_var = Some(id);
            }

            if name.contains("capacity") {
                capacity_var = Some(id);
            }
            if name.contains("load") {
                load_var = Some(id);
            }
        }

        if load_var.is_none() {
            'outer: for op in task.get_operators() {
                if op.name().starts_with("visit-via-depot") {
                    for assign in op.assignment_effects() {
                        load_var = Some(assign.affected_var_id());
                        break 'outer;
                    }
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values().to_vec();
        let mut demand_by_node = vec![0.0; n_nodes];

        if let Some(load_id) = load_var {
            for op in task.get_operators() {
                if let Some(parsed) = parse_operator_name(op.name()) {
                    if parsed.0 == "via" {
                        if let Some(&to_node) = node_id_by_name.get(parsed.2) {
                            for assign in op.assignment_effects() {
                                if assign.affected_var_id() == load_id {
                                    let rhs = assign.var_id();
                                    if rhs < initial_numeric.len() {
                                        let value = initial_numeric[rhs];
                                        if value.is_finite() && value >= 0.0 {
                                            demand_by_node[to_node] = value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let capacity = capacity_var
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|v| v.is_finite() && *v > 0.0);
        let capacity = if load_var.is_some() && demand_by_node.iter().any(|d| *d > 0.0) {
            capacity
        } else {
            None
        };

        let max_vehicles = max_vehicles_var
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|v| v.is_finite() && *v > 0.0);

        let scratch = RefCell::new(ScratchBuffers {
            propositional: vec![0; task.variables().len()],
            numeric: vec![0.0; task.numeric_variables().len()],
            unvisited: Vec::new(),
        });

        Ok(Self {
            depot_id: 0,
            customer_ids,
            visited_facts,
            loc_facts,
            direct,
            via,
            demand_by_node,
            load_var,
            vehicles_var,
            capacity,
            max_vehicles,
            avg_depot_round_trip,
            scratch,
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

        let mut scratch = self.scratch.borrow_mut();

        {
            let props = &mut scratch.propositional;
            let _ = state.fill_state(registry, props);
        }
        {
            let nums = &mut scratch.numeric;
            registry
                .view(state)
                .fill_numeric(nums)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!(
                        "LLM heuristic failed to read numeric state: {err:?}"
                    ))
                })?;
        }

        let props = &scratch.propositional;
        let nums = &scratch.numeric;

        let mut current = self.depot_id;
        for (node_id, loc_ref) in self.loc_facts.iter().enumerate() {
            if let Some(loc_ref) = loc_ref {
                if props[loc_ref.var_id] == loc_ref.positive_value {
                    current = node_id;
                    break;
                }
            }
        }

        scratch.unvisited.clear();
        for (ci, &node_id) in self.customer_ids.iter().enumerate() {
            let visited_ref = &self.visited_facts[ci];
            if props[visited_ref.var_id] != visited_ref.positive_value {
                scratch.unvisited.push(node_id);
            }
        }

        if scratch.unvisited.is_empty() {
            let h = if current == self.depot_id {
                0.0
            } else {
                self.direct[current][self.depot_id]
            };
            return Ok(h.max(0.0));
        }

        let mut total = 0.0;
        let mut prev = current;
        let mut load = if current == self.depot_id {
            0.0
        } else if let Some(load_id) = self.load_var {
            nums[load_id].max(0.0)
        } else {
            0.0
        };

        let mut vehicles_used = if let Some(vehicles_id) = self.vehicles_var {
            nums[vehicles_id].max(1.0)
        } else {
            1.0
        };

        while !scratch.unvisited.is_empty() {
            let mut best_feasible: Option<(usize, f64)> = None;
            let mut best_via: Option<(usize, f64)> = None;
            let mut best_any: Option<(usize, f64)> = None;

            for i in 0..scratch.unvisited.len() {
                let next = scratch.unvisited[i];
                let demand = self.demand_by_node[next];
                let direct_cost = self.direct[prev][next];

                if best_any.map_or(true, |(_, c)| direct_cost < c) {
                    best_any = Some((i, direct_cost));
                }

                let feasible = match self.capacity {
                    Some(cap) => load + demand <= cap,
                    None => true,
                };

                if feasible {
                    if best_feasible.map_or(true, |(_, c)| direct_cost < c) {
                        best_feasible = Some((i, direct_cost));
                    }
                } else if prev != self.depot_id {
                    let via_cost = self.via[prev][next];
                    if best_via.map_or(true, |(_, c)| via_cost < c) {
                        best_via = Some((i, via_cost));
                    }
                }
            }

            let next;
            if let Some((idx, cost)) = best_feasible {
                next = scratch.unvisited.swap_remove(idx);
                total += cost;
                load += self.demand_by_node[next];
            } else if let Some((idx, cost)) = best_via {
                next = scratch.unvisited.swap_remove(idx);
                total += cost;
                load = self.demand_by_node[next];
                vehicles_used += 1.0;
            } else if let Some((idx, cost)) = best_any {
                next = scratch.unvisited.swap_remove(idx);
                total += cost;
                load += self.demand_by_node[next];
                if self.capacity.is_some() {
                    vehicles_used += 1.0;
                    total += self.avg_depot_round_trip;
                }
            } else {
                break;
            }

            prev = next;
        }

        total += self.direct[prev][self.depot_id];

        if let Some(max_vehicles) = self.max_vehicles {
            let current_vehicles = if let Some(vehicles_id) = self.vehicles_var {
                nums[vehicles_id].max(1.0)
            } else {
                1.0
            };
            let used_extra = (vehicles_used - current_vehicles).max(0.0);
            let allowed_extra = (max_vehicles - current_vehicles).max(0.0);
            if used_extra > allowed_extra {
                total += (used_extra - allowed_extra) * self.avg_depot_round_trip;
            }
        }

        let h = total.max(0.0);
        if h.is_finite() {
            Ok(h)
        } else {
            Ok(0.0)
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
