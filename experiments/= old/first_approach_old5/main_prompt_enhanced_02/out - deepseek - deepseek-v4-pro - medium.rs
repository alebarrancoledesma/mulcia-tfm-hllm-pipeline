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
use std::collections::{BTreeSet, HashMap};

const LARGE_PENALTY: f64 = 10_000_000.0;

fn base_pne_name(raw: &str) -> &str {
    let t = raw.trim();
    t.strip_prefix("PNE ").unwrap_or(t)
}

fn parse_simple_atom(fact_name: &str, predicate: &str) -> Option<String> {
    let s = fact_name.trim();
    let needle = format!("Atom {} (", predicate);
    let start = s.find(&needle)?;
    let rest = &s[start + needle.len()..];
    if !rest.ends_with(')') {
        return None;
    }
    let inner = &rest[..rest.len() - 1];
    if inner.is_empty() || inner.contains(' ') || inner.contains(',') {
        return None;
    }
    Some(inner.to_string())
}

fn direct_cost_static(
    from: usize,
    to: usize,
    depot_idx: usize,
    direct: &[Vec<f64>],
) -> f64 {
    let mut c = direct[from][to];
    if c.is_finite() {
        return c;
    }
    if from != depot_idx && to != depot_idx {
        let a = direct[from][depot_idx];
        let b = direct[depot_idx][to];
        if a.is_finite() && b.is_finite() {
            c = a + b;
        }
    }
    c
}

fn via_cost_static(
    from: usize,
    to: usize,
    depot_idx: usize,
    direct: &[Vec<f64>],
    via: &[Vec<f64>],
) -> f64 {
    let mut c = via[from][to];
    if c.is_finite() {
        return c;
    }
    if from != depot_idx && to != depot_idx {
        let a = direct[from][depot_idx];
        let b = direct[depot_idx][to];
        if a.is_finite() && b.is_finite() {
            c = a + b;
        }
    }
    c
}

fn effective_cost(
    i: usize,
    j: usize,
    depot_idx: usize,
    direct: &[Vec<f64>],
    via: &[Vec<f64>],
) -> f64 {
    let mut c = direct[i][j].min(via[i][j]);
    if i != depot_idx && j != depot_idx {
        let a = direct[i][depot_idx];
        let b = direct[depot_idx][j];
        if a.is_finite() && b.is_finite() {
            c = c.min(a + b);
        }
    }
    c
}

#[allow(clippy::too_many_arguments)]
fn choose_next_static(
    cur: usize,
    route_load: f64,
    vehicles_used: f64,
    capacity: f64,
    max_vehicles: f64,
    depot_idx: usize,
    demands: &[f64],
    direct: &[Vec<f64>],
    via: &[Vec<f64>],
    customer_indices: &[usize],
    is_customer: &[bool],
    unvisited: &[bool],
    nearest_from: &[Vec<usize>],
) -> Option<(usize, f64, bool)> {
    for &nb in &nearest_from[cur] {
        if !is_customer[nb] || !unvisited[nb] {
            continue;
        }
        let dem = demands[nb];
        let direct_ok = route_load + dem <= capacity + 1e-9;
        let via_ok = cur != depot_idx
            && vehicles_used + 1.0 <= max_vehicles + 1e-9
            && dem <= capacity + 1e-9;

        if direct_ok {
            let dc = direct_cost_static(cur, nb, depot_idx, direct);
            if dc.is_finite() {
                return Some((nb, dc, false));
            }
        }
        if via_ok {
            let vc = via_cost_static(cur, nb, depot_idx, direct, via);
            if vc.is_finite() {
                return Some((nb, vc, true));
            }
        }
    }

    let mut best: Option<(usize, f64, bool)> = None;
    for &nb in customer_indices {
        if !unvisited[nb] {
            continue;
        }
        let dem = demands[nb];
        let direct_ok = route_load + dem <= capacity + 1e-9;
        let via_ok = cur != depot_idx
            && vehicles_used + 1.0 <= max_vehicles + 1e-9
            && dem <= capacity + 1e-9;

        if direct_ok {
            let dc = direct_cost_static(cur, nb, depot_idx, direct);
            if dc.is_finite() && best.as_ref().map_or(true, |b| dc < b.1) {
                best = Some((nb, dc, false));
            }
        }
        if via_ok {
            let vc = via_cost_static(cur, nb, depot_idx, direct, via);
            if vc.is_finite() && best.as_ref().map_or(true, |b| vc < b.1) {
                best = Some((nb, vc, true));
            }
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn simulate_routing_static(
    start_cur: usize,
    start_load: f64,
    start_vehicles: f64,
    unvisited: &mut [bool],
    mut count: usize,
    depot_idx: usize,
    capacity: f64,
    max_vehicles: f64,
    demands: &[f64],
    direct: &[Vec<f64>],
    via: &[Vec<f64>],
    customer_indices: &[usize],
    is_customer: &[bool],
    nearest_from: &[Vec<usize>],
) -> (f64, bool) {
    let mut cur = start_cur;
    let mut route_load = start_load;
    let mut vehicles = start_vehicles;
    let mut cost = 0.0;
    let mut dead_end = false;

    while count > 0 {
        let chosen = choose_next_static(
            cur,
            route_load,
            vehicles,
            capacity,
            max_vehicles,
            depot_idx,
            demands,
            direct,
            via,
            customer_indices,
            is_customer,
            unvisited,
            nearest_from,
        );

        match chosen {
            Some((nb, move_cost, used_via)) => {
                cost += move_cost;
                if used_via {
                    vehicles += 1.0;
                    route_load = demands[nb];
                } else {
                    route_load += demands[nb];
                }
                unvisited[nb] = false;
                cur = nb;
                count -= 1;
            }
            None => {
                dead_end = true;
                break;
            }
        }
    }

    if !dead_end && cur != depot_idx {
        let rc = direct[cur][depot_idx];
        if rc.is_finite() {
            cost += rc;
        } else {
            dead_end = true;
        }
    }

    (cost, dead_end)
}

struct LlmHeuristic {
    place_count: usize,
    depot_idx: usize,
    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,
    loc_facts: Vec<(usize, usize)>,
    visited_facts: Vec<(usize, usize)>,
    customer_indices: Vec<usize>,
    is_customer: Vec<bool>,
    demands: Vec<f64>,
    direct_cost: Vec<Vec<f64>>,
    via_depot_cost: Vec<Vec<f64>>,
    nearest_from: Vec<Vec<usize>>,
    scratch_numeric: RefCell<Vec<f64>>,
    scratch_props: RefCell<Vec<usize>>,
    scratch_unvisited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let mut loc_atoms: Vec<(String, usize, usize)> = Vec::new();
        let mut visited_atoms: Vec<(String, usize, usize)> = Vec::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                if let Some(obj) = parse_simple_atom(&fact_name, "loc") {
                    loc_atoms.push((obj, var_id, value));
                }
                if let Some(obj) = parse_simple_atom(&fact_name, "visited") {
                    visited_atoms.push((obj, var_id, value));
                }
            }
        }

        let customer_set: BTreeSet<String> = visited_atoms.iter().map(|x| x.0.clone()).collect();
        let mut place_set: BTreeSet<String> = BTreeSet::new();
        for (name, _, _) in &loc_atoms {
            place_set.insert(name.clone());
        }
        for name in &customer_set {
            place_set.insert(name.clone());
        }

        let mut parsed_ops: Vec<(u8, String, Option<String>, f64)> = Vec::new();
        for op in task.get_operators() {
            let cost = metric_operator_cost_from_initial_values(task, op);
            let tokens: Vec<&str> = op.name().split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            match tokens[0] {
                "visit" if tokens.len() >= 3 => {
                    place_set.insert(tokens[1].to_string());
                    place_set.insert(tokens[2].to_string());
                    parsed_ops.push((0, tokens[1].to_string(), Some(tokens[2].to_string()), cost));
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    place_set.insert(tokens[1].to_string());
                    place_set.insert(tokens[2].to_string());
                    parsed_ops.push((1, tokens[1].to_string(), Some(tokens[2].to_string()), cost));
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    place_set.insert(tokens[1].to_string());
                    parsed_ops.push((2, tokens[1].to_string(), None, cost));
                }
                _ => {}
            }
        }

        let place_names: Vec<String> = place_set.into_iter().collect();
        let mut place_idx: HashMap<String, usize> = HashMap::new();
        for (i, name) in place_names.iter().enumerate() {
            place_idx.insert(name.clone(), i);
        }
        let p = place_names.len();

        let depot_idx: usize = {
            let mut found = None;
            for (name, _, _) in &loc_atoms {
                if !customer_set.contains(name) {
                    if let Some(&idx) = place_idx.get(name.as_str()) {
                        found = Some(idx);
                        break;
                    }
                }
            }
            found.ok_or_else(|| HeuristicBuildError::from("no depot loc fact found".to_string()))?
        };

        let initial_numeric = task.get_initial_numeric_state_values().to_vec();
        let mut numeric_ids: HashMap<String, usize> = HashMap::new();
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let base = base_pne_name(var.name()).to_string();
            numeric_ids.insert(base, id);
        }

        let load_id = *numeric_ids
            .get("load()")
            .ok_or_else(|| HeuristicBuildError::from("load() numeric variable not found".to_string()))?;
        let vehicles_id = *numeric_ids
            .get("vehicles()")
            .ok_or_else(|| HeuristicBuildError::from("vehicles() numeric variable not found".to_string()))?;

        let capacity = numeric_ids
            .get("capacity()")
            .map(|&id| initial_numeric[id])
            .unwrap_or(f64::MAX);
        let max_vehicles = numeric_ids
            .get("max_vehicles()")
            .map(|&id| initial_numeric[id])
            .unwrap_or(f64::MAX);

        let mut direct_cost = vec![vec![f64::INFINITY; p]; p];
        let mut via_depot_cost = vec![vec![f64::INFINITY; p]; p];

        for (kind, from, to, cost) in parsed_ops {
            let i = place_idx[&from];
            match kind {
                0 => {
                    let j = place_idx[&to.unwrap()];
                    direct_cost[i][j] = direct_cost[i][j].min(cost);
                }
                1 => {
                    let j = place_idx[&to.unwrap()];
                    via_depot_cost[i][j] = via_depot_cost[i][j].min(cost);
                }
                2 => {
                    direct_cost[i][depot_idx] = direct_cost[i][depot_idx].min(cost);
                }
                _ => {}
            }
        }

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let base = base_pne_name(var.name());
            if let Some(inner) = base
                .strip_prefix("travel-cost(")
                .and_then(|x| x.strip_suffix(')'))
            {
                let inner = inner.replace(',', " ");
                let parts: Vec<&str> = inner.split_whitespace().collect();
                if parts.len() == 2 {
                    if let (Some(&i), Some(&j)) = (place_idx.get(parts[0]), place_idx.get(parts[1])) {
                        direct_cost[i][j] = direct_cost[i][j].min(initial_numeric[id]);
                    }
                }
            }
        }

        let customer_names: Vec<String> = customer_set.into_iter().collect();
        let mut customer_indices: Vec<usize> = Vec::with_capacity(customer_names.len());
        let mut visited_facts: Vec<(usize, usize)> = Vec::with_capacity(customer_names.len());
        let mut demands = vec![0.0; p];
        let mut is_customer = vec![false; p];

        for cname in &customer_names {
            let pi = place_idx[cname.as_str()];
            customer_indices.push(pi);
            is_customer[pi] = true;

            let mut ff: Option<(usize, usize)> = None;
            for (name, var_id, value) in &visited_atoms {
                if name == cname {
                    ff = Some((*var_id, *value));
                    break;
                }
            }
            let ff = ff.ok_or_else(|| {
                HeuristicBuildError::from(format!("visited fact not found for {}", cname))
            })?;
            visited_facts.push(ff);

            let demand_key = format!("demand({})", cname);
            if let Some(&did) = numeric_ids.get(&demand_key) {
                demands[pi] = initial_numeric[did];
            }
        }

        let mut loc_facts = vec![(usize::MAX, usize::MAX); p];
        for (name, var_id, value) in &loc_atoms {
            if let Some(&pi) = place_idx.get(name.as_str()) {
                loc_facts[pi] = (*var_id, *value);
            }
        }
        for pi in 0..p {
            if loc_facts[pi] == (usize::MAX, usize::MAX) {
                return Err(HeuristicBuildError::from(format!(
                    "missing loc fact for place {}",
                    place_names[pi]
                )));
            }
        }

        let mut nearest_from: Vec<Vec<usize>> = Vec::with_capacity(p);
        for i in 0..p {
            let mut neigh: Vec<usize> = (0..p).filter(|&j| j != i).collect();
            neigh.sort_by(|&a, &b| {
                let ca = effective_cost(i, a, depot_idx, &direct_cost, &via_depot_cost);
                let cb = effective_cost(i, b, depot_idx, &direct_cost, &via_depot_cost);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });
            nearest_from.push(neigh);
        }

        if customer_indices.len() > 1 {
            for &pi in &customer_indices {
                if direct_cost[depot_idx][pi].is_infinite() {
                    return Err(HeuristicBuildError::from(format!(
                        "missing depot-to-customer cost for {}",
                        place_names[pi]
                    )));
                }
                let mut usable = false;
                for &pj in &customer_indices {
                    if pi != pj
                        && effective_cost(pi, pj, depot_idx, &direct_cost, &via_depot_cost).is_finite()
                    {
                        usable = true;
                        break;
                    }
                }
                if !usable {
                    return Err(HeuristicBuildError::from(format!(
                        "customer {} appears disconnected",
                        place_names[pi]
                    )));
                }
            }
        } else if let Some(&pi) = customer_indices.first() {
            if direct_cost[depot_idx][pi].is_infinite() {
                return Err(HeuristicBuildError::from(format!(
                    "missing depot-to-customer cost for {}",
                    place_names[pi]
                )));
            }
        }

        Ok(Self {
            place_count: p,
            depot_idx,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            loc_facts,
            visited_facts,
            customer_indices,
            is_customer,
            demands,
            direct_cost,
            via_depot_cost,
            nearest_from,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_props: RefCell::new(vec![0; task.variables().len()]),
            scratch_unvisited: RefCell::new(vec![false; p]),
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

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let num_numeric = task.numeric_variables().len();
        let num_props = task.variables().len();

        let h = {
            let mut nv = self.scratch_numeric.borrow_mut();
            nv.resize(num_numeric, 0.0);
            registry
                .view(state)
                .fill_numeric(&mut nv)
                .map_err(|e| EvaluationError::ComputationFailed(format!("{:?}", e)))?;
            let load = nv[self.load_id];
            let vehicles = nv[self.vehicles_id];
            drop(nv);

            let mut pv = self.scratch_props.borrow_mut();
            pv.resize(num_props, 0);
            state
                .fill_state(registry, &mut pv)
                .map_err(|e| EvaluationError::ComputationFailed(format!("{:?}", e)))?;

            let mut cur = self.depot_idx;
            for (pi, &(var_id, value)) in self.loc_facts.iter().enumerate() {
                if pv[var_id] == value {
                    cur = pi;
                    break;
                }
            }

            let mut uv = self.scratch_unvisited.borrow_mut();
            uv.resize(self.place_count, false);
            let mut remaining_demand = 0.0;
            let mut unvisited_count = 0usize;

            for (ci, &(var_id, value)) in self.visited_facts.iter().enumerate() {
                let pi = self.customer_indices[ci];
                let is_visited = pv[var_id] == value;
                if !is_visited {
                    uv[pi] = true;
                    unvisited_count += 1;
                    remaining_demand += self.demands[pi];
                }
            }
            drop(pv);

            let mut h_resource = 0.0;
            if self.capacity.is_finite() {
                if unvisited_count > 0 {
                    for &pi in &self.customer_indices {
                        if uv[pi] && self.demands[pi] > self.capacity + 1e-9 {
                            h_resource += LARGE_PENALTY;
                            break;
                        }
                    }
                }
                if remaining_demand > 0.0 {
                    let residual = (self.capacity - load).max(0.0);
                    let mut available = residual;
                    if self.max_vehicles.is_finite() {
                        let remaining_vehicles = (self.max_vehicles - vehicles).max(0.0);
                        available += remaining_vehicles * self.capacity;
                    } else {
                        available = f64::MAX;
                    }
                    if remaining_demand > available + 1e-9 {
                        h_resource += LARGE_PENALTY + 100.0 * (remaining_demand - available);
                    }
                }
            }
            if self.max_vehicles.is_finite() && vehicles > self.max_vehicles + 1e-9 {
                h_resource += LARGE_PENALTY;
            }

            let (routing_cost, dead_end) = simulate_routing_static(
                cur,
                load,
                vehicles,
                uv.as_mut_slice(),
                unvisited_count,
                self.depot_idx,
                self.capacity,
                self.max_vehicles,
                &self.demands,
                &self.direct_cost,
                &self.via_depot_cost,
                &self.customer_indices,
                &self.is_customer,
                &self.nearest_from,
            );

            let mut h = routing_cost + h_resource;
            if dead_end {
                h += LARGE_PENALTY;
            }
            h
        };

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
                .into(),
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
