use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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

const INF: f64 = 1e12;

#[derive(Clone, Copy)]
struct PropFactRef {
    var_id: usize,
    positive_value: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
}

struct LlmHeuristic {
    num_nodes: usize,
    visited_refs: Vec<Option<PropFactRef>>,
    loc_refs: Vec<(usize, PropFactRef)>,
    demand: Vec<f64>,
    direct: Vec<Vec<f64>>,
    via_depot: Vec<Vec<f64>>,
    return_cost: Vec<f64>,
    depot_out: Vec<f64>,
    load_var: Option<usize>,
    vehicles_var: Option<usize>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    scratch: RefCell<ScratchBuffers>,
}

fn extract_atom_argument(fact_name: &str, predicate: &str) -> Option<String> {
    let prefix = format!("Atom {predicate}(");
    if fact_name.starts_with(&prefix) && fact_name.ends_with(')') {
        let inner = &fact_name[prefix.len()..fact_name.len() - 1];
        if !inner.is_empty() {
            return Some(inner.to_string());
        }
    }
    None
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut visited_fact_map: HashMap<String, PropFactRef> = HashMap::new();
        let mut loc_fact_map: HashMap<String, PropFactRef> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                let fact_name = fact_name.trim();

                if let Some(customer) = extract_atom_argument(fact_name, "visited") {
                    visited_fact_map.insert(customer, PropFactRef { var_id, positive_value: value });
                } else if let Some(place) = extract_atom_argument(fact_name, "loc") {
                    loc_fact_map.insert(place, PropFactRef { var_id, positive_value: value });
                }
            }
        }

        let mut customer_names: Vec<String> = visited_fact_map.keys().cloned().collect();
        customer_names.sort();

        let mut customer_to_id: HashMap<String, usize> = HashMap::new();
        for (i, name) in customer_names.iter().enumerate() {
            customer_to_id.insert(name.clone(), i + 1);
        }

        let mut depot_name: Option<String> = None;
        for place_name in loc_fact_map.keys() {
            if !customer_to_id.contains_key(place_name.as_str()) {
                depot_name = Some(place_name.clone());
                break;
            }
        }
        let depot_name = depot_name.unwrap_or_else(|| "d1".to_string());

        let mut place_to_node: HashMap<String, usize> = HashMap::new();
        place_to_node.insert(depot_name.clone(), 0);
        for (i, name) in customer_names.iter().enumerate() {
            place_to_node.insert(name.clone(), i + 1);
        }

        let num_nodes = customer_names.len() + 1;
        let mut visited_refs: Vec<Option<PropFactRef>> = vec![None; num_nodes];
        for (i, name) in customer_names.iter().enumerate() {
            visited_refs[i + 1] = visited_fact_map.remove(name);
        }

        let mut loc_refs = Vec::new();
        for (place_name, fact_ref) in loc_fact_map {
            if let Some(&node_id) = place_to_node.get(place_name.as_str()) {
                loc_refs.push((node_id, fact_ref));
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();

        let mut visit_affected: HashSet<usize> = HashSet::new();
        let mut via_affected: HashSet<usize> = HashSet::new();
        let mut return_affected: HashSet<usize> = HashSet::new();
        let mut referenced_vars: HashSet<usize> = HashSet::new();

        for op in task.get_operators() {
            let op_name = op.name().to_string();
            let parts: Vec<&str> = op_name.split_whitespace().collect();
            let action = parts.first().copied();

            if action == Some("visit") {
                for assign in op.assignment_effects() {
                    visit_affected.insert(assign.affected_var_id());
                    referenced_vars.insert(assign.var_id());
                }
            } else if action == Some("visit-via-depot") {
                for assign in op.assignment_effects() {
                    via_affected.insert(assign.affected_var_id());
                    referenced_vars.insert(assign.var_id());
                }
            } else if action == Some("return-to-depot") {
                for assign in op.assignment_effects() {
                    return_affected.insert(assign.affected_var_id());
                    referenced_vars.insert(assign.var_id());
                }
            }
        }

        let total_cost_id = visit_affected
            .intersection(&via_affected)
            .cloned()
            .find(|id| return_affected.contains(id));

        let load_id = visit_affected
            .intersection(&via_affected)
            .cloned()
            .find(|id| !return_affected.contains(id));

        let vehicles_id = via_affected
            .iter()
            .cloned()
            .find(|id| !visit_affected.contains(id) && !return_affected.contains(id));

        let mut affected_vars: HashSet<usize> = HashSet::new();
        for id in visit_affected.iter().chain(via_affected.iter()).chain(return_affected.iter()) {
            affected_vars.insert(*id);
        }

        let mut static_unreferenced: Vec<usize> = Vec::new();
        for id in 0..initial_numeric.len() {
            if !affected_vars.contains(&id) && !referenced_vars.contains(&id) {
                static_unreferenced.push(id);
            }
        }

        let mut capacity = None;
        let mut max_vehicles = None;
        if static_unreferenced.len() == 2 {
            let a = static_unreferenced[0];
            let b = static_unreferenced[1];
            let (cap_id, max_id) = if initial_numeric[a] >= initial_numeric[b] {
                (a, b)
            } else {
                (b, a)
            };
            capacity = Some(initial_numeric[cap_id].abs().max(0.0));
            max_vehicles = Some(initial_numeric[max_id].abs().max(0.0));
        }

        let mut demand = vec![0.0; num_nodes];
        let mut direct = vec![vec![INF; num_nodes]; num_nodes];
        let mut via_depot = vec![vec![INF; num_nodes]; num_nodes];
        let mut return_cost = vec![INF; num_nodes];
        let mut depot_out = vec![INF; num_nodes];

        for op in task.get_operators() {
            let op_name = op.name().to_string();
            let parts: Vec<&str> = op_name.split_whitespace().collect();
            let action = parts.first().copied();

            let mut cost = INF;
            let mut demand_value = 0.0;

            for assign in op.assignment_effects() {
                let aff = assign.affected_var_id();
                if total_cost_id == Some(aff) {
                    let v = initial_numeric.get(assign.var_id()).copied().unwrap_or(0.0).abs();
                    if v < cost {
                        cost = v;
                    }
                }
                if load_id == Some(aff) {
                    demand_value = initial_numeric.get(assign.var_id()).copied().unwrap_or(0.0).abs();
                }
            }

            if action == Some("visit") && parts.len() == 3 {
                let from_name = parts[1];
                let to_name = parts[2];
                if let (Some(&from_id), Some(&to_id)) = (place_to_node.get(from_name), place_to_node.get(to_name)) {
                    if cost < INF && to_id != 0 {
                        if from_id == 0 {
                            depot_out[to_id] = depot_out[to_id].min(cost);
                        }
                        direct[from_id][to_id] = direct[from_id][to_id].min(cost);
                    }
                    if demand_value > 0.0 && to_id != 0 {
                        demand[to_id] = demand_value;
                    }
                }
            } else if action == Some("visit-via-depot") && parts.len() == 3 {
                let from_name = parts[1];
                let to_name = parts[2];
                if let (Some(&from_id), Some(&to_id)) = (place_to_node.get(from_name), place_to_node.get(to_name)) {
                    if cost < INF && from_id != 0 && to_id != 0 {
                        via_depot[from_id][to_id] = via_depot[from_id][to_id].min(cost);
                    }
                    if demand_value > 0.0 && to_id != 0 {
                        demand[to_id] = demand_value;
                    }
                }
            } else if action == Some("return-to-depot") && parts.len() == 2 {
                let from_name = parts[1];
                if let Some(&from_id) = place_to_node.get(from_name) {
                    if cost < INF && from_id != 0 {
                        return_cost[from_id] = return_cost[from_id].min(cost);
                        direct[from_id][0] = direct[from_id][0].min(cost);
                    }
                }
            }
        }

        let scratch = RefCell::new(ScratchBuffers {
            propositional: Vec::with_capacity(task.variables().len()),
            numeric: Vec::with_capacity(initial_numeric.len()),
        });

        Ok(Self {
            num_nodes,
            visited_refs,
            loc_refs,
            demand,
            direct,
            via_depot,
            return_cost,
            depot_out,
            load_var: load_id,
            vehicles_var: vehicles_id,
            capacity,
            max_vehicles,
            scratch,
        })
    }

    fn edge_cost(&self, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }

        let mut best = INF;

        if let Some(v) = self.direct.get(a).and_then(|row| row.get(b)).copied() {
            best = best.min(v);
        }
        if let Some(v) = self.direct.get(b).and_then(|row| row.get(a)).copied() {
            best = best.min(v);
        }
        if let Some(v) = self.via_depot.get(a).and_then(|row| row.get(b)).copied() {
            best = best.min(v);
        }
        if let Some(v) = self.via_depot.get(b).and_then(|row| row.get(a)).copied() {
            best = best.min(v);
        }

        if a != 0 && b != 0 {
            let ra = self.return_cost.get(a).copied().unwrap_or(INF);
            let rb = self.return_cost.get(b).copied().unwrap_or(INF);
            let oa = self.depot_out.get(a).copied().unwrap_or(INF);
            let ob = self.depot_out.get(b).copied().unwrap_or(INF);
            if ra < INF && ob < INF {
                best = best.min(ra + ob);
            }
            if rb < INF && oa < INF {
                best = best.min(rb + oa);
            }
        } else {
            let c = if a == 0 { b } else { a };
            if let Some(v) = self.depot_out.get(c).copied() {
                best = best.min(v);
            }
            if let Some(v) = self.return_cost.get(c).copied() {
                best = best.min(v);
            }
        }

        best
    }

    fn new_vehicle_cost(&self, from: usize, to: usize) -> f64 {
        let mut best = self
            .via_depot
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(INF);

        let r = self.return_cost.get(from).copied().unwrap_or(INF);
        let o = self.depot_out.get(to).copied().unwrap_or(INF);
        if r < INF && o < INF {
            best = best.min(r + o);
        }

        best
    }

    fn mst_estimate(&self, start: usize, unvisited: &[bool]) -> f64 {
        let mut nodes = Vec::with_capacity(self.num_nodes + 1);
        nodes.push(0usize);
        if start != 0 {
            nodes.push(start);
        }
        for c in 1..self.num_nodes {
            if unvisited[c] {
                nodes.push(c);
            }
        }

        if nodes.len() <= 1 {
            return 0.0;
        }

        let n = nodes.len();
        let mut in_tree = vec![false; n];
        let mut min_edge = vec![INF; n];
        min_edge[0] = 0.0;
        let mut total = 0.0;

        for _ in 0..n {
            let mut u = None;
            for i in 0..n {
                if !in_tree[i] && (u.is_none() || min_edge[i] < min_edge[u.unwrap()]) {
                    u = Some(i);
                }
            }

            let u = match u {
                Some(x) => x,
                None => break,
            };

            in_tree[u] = true;
            total += min_edge[u];

            for v in 0..n {
                if !in_tree[v] {
                    let w = self.edge_cost(nodes[u], nodes[v]);
                    if w < min_edge[v] {
                        min_edge[v] = w;
                    }
                }
            }
        }

        total
    }

    fn greedy_estimate(
        &self,
        start: usize,
        mut load: f64,
        mut vehicle_count: f64,
        remaining: &mut [bool],
    ) -> f64 {
        let mut current = start;
        let mut total = 0.0;
        let mut left = remaining.iter().filter(|&&b| b).count();

        while left > 0 {
            let mut best_cost = INF;
            let mut best: Option<(usize, bool, f64, f64)> = None;

            for c in 1..self.num_nodes {
                if !remaining[c] {
                    continue;
                }

                let demand_value = self.demand.get(c).copied().unwrap_or(0.0).max(0.0);

                if current == 0 {
                    let cost = self
                        .direct
                        .get(0)
                        .and_then(|row| row.get(c))
                        .copied()
                        .unwrap_or(INF);
                    if cost >= INF {
                        continue;
                    }

                    let new_load = load + demand_value;
                    let feasible = self.capacity.map_or(true, |cap| new_load <= cap + 1e-9);
                    if feasible && cost < best_cost {
                        best_cost = cost;
                        best = Some((c, false, new_load, vehicle_count));
                    }
                } else {
                    if self.capacity.map_or(true, |cap| load + demand_value <= cap + 1e-9) {
                        let cost = self
                            .direct
                            .get(current)
                            .and_then(|row| row.get(c))
                            .copied()
                            .unwrap_or(INF);
                        if cost < INF && cost < best_cost {
                            best_cost = cost;
                            best = Some((c, false, load + demand_value, vehicle_count));
                        }
                    }

                    if self.max_vehicles.map_or(true, |mv| vehicle_count + 1.0 <= mv + 1e-9) {
                        let cost = self.new_vehicle_cost(current, c);
                        if cost < INF && cost < best_cost {
                            best_cost = cost;
                            best = Some((c, true, demand_value, vehicle_count + 1.0));
                        }
                    }
                }
            }

            if let Some((next, _, new_load, new_vehicle_count)) = best {
                total += best_cost;
                remaining[next] = false;
                current = next;
                load = new_load;
                vehicle_count = new_vehicle_count;
                left -= 1;
            } else {
                break;
            }
        }

        if current != 0 {
            let r = self
                .return_cost
                .get(current)
                .copied()
                .unwrap_or(INF)
                .min(self.direct.get(current).and_then(|row| row.get(0)).copied().unwrap_or(INF));
            if r < INF {
                total += r;
            }
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

        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut scratch = self.scratch.borrow_mut();
        scratch.propositional.clear();
        state.fill_state(registry, &mut scratch.propositional);

        scratch.numeric.clear();
        registry
            .view(state)
            .fill_numeric(&mut scratch.numeric)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!(
                    "LLM heuristic failed to read numeric state: {err:?}"
                ))
            })?;

        let prop_vals = &scratch.propositional;
        let num_vals = &scratch.numeric;

        let mut unvisited = vec![false; self.num_nodes];
        let mut any_unvisited = false;
        for c in 1..self.num_nodes {
            let is_unvisited = match &self.visited_refs[c] {
                Some(f) => {
                    prop_vals.get(f.var_id).copied().unwrap_or(f.positive_value) != f.positive_value
                }
                None => true,
            };
            if is_unvisited {
                any_unvisited = true;
            }
            unvisited[c] = is_unvisited;
        }

        let current = self
            .loc_refs
            .iter()
            .find(|(_, f)| {
                prop_vals.get(f.var_id).copied().unwrap_or(f.positive_value) == f.positive_value
            })
            .map(|(node, _)| *node)
            .unwrap_or(0);

        let load = self
            .load_var
            .and_then(|id| num_vals.get(id))
            .copied()
            .unwrap_or(0.0)
            .max(0.0);

        let vehicles = self
            .vehicles_var
            .and_then(|id| num_vals.get(id))
            .copied()
            .unwrap_or(1.0)
            .max(0.0);

        drop(scratch);

        if !any_unvisited {
            if current != 0 {
                let r = self
                    .return_cost
                    .get(current)
                    .copied()
                    .unwrap_or(INF)
                    .min(self.direct.get(current).and_then(|row| row.get(0)).copied().unwrap_or(INF));
                if r < INF {
                    return Ok(r.max(0.0));
                }
                return Ok(0.0);
            }
            return Ok(0.0);
        }

        let mst = self.mst_estimate(current, &unvisited);
        let mut h = if mst.is_finite() { mst.max(0.0) } else { 0.0 };

        if self.load_var.is_some() || self.capacity.is_some() {
            let mut route_remaining = unvisited.clone();
            let greedy = self.greedy_estimate(current, load, vehicles, &mut route_remaining);
            if greedy.is_finite() && greedy > h {
                h = greedy;
            }
        }

        if h == 0.0 {
            let mut current_fallback = current;
            let mut remaining = unvisited.clone();
            let mut left = remaining.iter().filter(|&&b| b).count();
            let mut total = 0.0;

            while left > 0 {
                let mut best_cost = INF;
                let mut best_next = 0;
                for c in 1..self.num_nodes {
                    if remaining[c] {
                        let w = self.edge_cost(current_fallback, c);
                        if w < best_cost {
                            best_cost = w;
                            best_next = c;
                        }
                    }
                }
                if best_cost >= INF {
                    break;
                }
                total += best_cost;
                current_fallback = best_next;
                remaining[current_fallback] = false;
                left -= 1;
            }

            if current_fallback != 0 {
                let r = self.edge_cost(current_fallback, 0);
                if r < INF {
                    total += r;
                }
            }

            h = total;
        }

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
