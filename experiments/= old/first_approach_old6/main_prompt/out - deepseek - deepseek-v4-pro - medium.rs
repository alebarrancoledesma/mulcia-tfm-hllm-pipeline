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

use std::cmp::Ordering;
use std::collections::HashMap;


struct LlmHeuristic {
    n_customers: usize,
    n_nodes: usize,
    depot_idx: usize,
    visited_facts: Vec<(usize, usize)>,
    loc_var_id: Option<usize>,
    loc_value_to_place: Vec<Option<usize>>,
    loc_fallback: Vec<(usize, usize, usize)>,
    dist: Vec<f64>,
    via_dist: Vec<f64>,
    neighbors: Vec<Vec<usize>>,
    demands: Vec<f64>,
    capacity: f64,
    max_vehicles: f64,
    load_var: usize,
    vehicles_var: usize,
}

const INF: f64 = 1e30;
const EPS: f64 = 1e-6;

fn parse_atom_arg(name: &str, prefix: &str) -> Option<String> {
    let rest = name.strip_prefix(prefix)?;
    let rest = rest.strip_suffix(")")?;
    Some(rest.to_string())
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        let variables = task.variables();
        let mut visited_raw: Vec<(String, usize, usize)> = Vec::new();
        let mut loc_raw: Vec<(String, usize, usize)> = Vec::new();

        for (var_id, var) in variables.iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact).to_string();
                let fact_name = fact_name.trim();
                if let Some(obj) = parse_atom_arg(fact_name, "Atom visited(") {
                    visited_raw.push((obj, var_id, value));
                } else if let Some(obj) = parse_atom_arg(fact_name, "Atom loc(") {
                    loc_raw.push((obj, var_id, value));
                }
            }
        }

        let mut customer_names: Vec<String> = Vec::new();
        for (name, _, _) in &visited_raw {
            if !customer_names.contains(name) {
                customer_names.push(name.clone());
            }
        }
        customer_names.sort();

        let n_customers = customer_names.len();
        let n_nodes = n_customers + 1;
        let depot_idx = 0usize;

        let depot_name = loc_raw
            .iter()
            .find(|(name, _, _)| !customer_names.contains(name))
            .map(|(name, _, _)| name.clone())
            .ok_or_else(|| HeuristicBuildError::from("CVRP heuristic: no depot loc fact found".to_string()))?;

        let mut place_to_idx: HashMap<String, usize> = HashMap::new();
        place_to_idx.insert(depot_name.clone(), depot_idx);
        for (i, name) in customer_names.iter().enumerate() {
            place_to_idx.insert(name.clone(), i + 1);
        }

        let mut visited_facts = vec![(usize::MAX, usize::MAX); n_nodes];
        for (name, var_id, value) in visited_raw {
            if let Some(&idx) = place_to_idx.get(&name) {
                if idx >= 1 && idx <= n_customers {
                    visited_facts[idx] = (var_id, value);
                }
            }
        }

        let loc_var_id = if loc_raw.is_empty() {
            None
        } else {
            let first_var = loc_raw[0].1;
            if loc_raw.iter().all(|(_, v, _)| *v == first_var) {
                Some(first_var)
            } else {
                None
            }
        };

        let mut loc_value_to_place: Vec<Option<usize>> = Vec::new();
        if let Some(var_id) = loc_var_id {
            loc_value_to_place = vec![None; variables[var_id].domain_size()];
        }

        let mut loc_fallback = Vec::new();
        for (name, var_id, value) in loc_raw {
            if let Some(&idx) = place_to_idx.get(&name) {
                loc_fallback.push((var_id, value, idx));
                if let Some(lvid) = loc_var_id {
                    if var_id == lvid && value < loc_value_to_place.len() {
                        loc_value_to_place[value] = Some(idx);
                    }
                }
            }
        }

        let mut load_var = usize::MAX;
        let mut vehicles_var = usize::MAX;
        let mut capacity = f64::MAX;
        let mut max_vehicles = f64::MAX;

        let numeric_variables = task.numeric_variables();
        let initial_numeric = task.get_initial_numeric_state_values();

        for (id, var) in numeric_variables.iter().enumerate() {
            let name = var.name().trim();
            if name == "load()" || name == "PNE load()" {
                load_var = id;
            } else if name == "vehicles()" || name == "PNE vehicles()" {
                vehicles_var = id;
            } else if name == "capacity()" || name == "PNE capacity()" {
                capacity = initial_numeric[id];
            } else if name == "max_vehicles()" || name == "PNE max_vehicles()" {
                max_vehicles = initial_numeric[id];
            }
        }

        let mut demands = vec![0.0; n_nodes];
        for (i, cust_name) in customer_names.iter().enumerate() {
            let customer_idx = i + 1;
            let exact1 = format!("demand({})", cust_name);
            let exact2 = format!("PNE demand({})", cust_name);
            for (id, var) in numeric_variables.iter().enumerate() {
                let name = var.name().trim();
                if name == exact1 || name == exact2 {
                    demands[customer_idx] = initial_numeric[id];
                    break;
                }
            }
        }

        let mut dist = vec![INF; n_nodes * n_nodes];
        for i in 0..n_nodes {
            dist[i * n_nodes + i] = 0.0;
        }
        let mut via_dist = vec![INF; n_nodes * n_nodes];

        let operators = task.get_operators();
        for op in operators.iter() {
            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() {
                continue;
            }
            let tokens: Vec<&str> = op.name().split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            match tokens[0] {
                "visit" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_to_idx.get(tokens[1]), place_to_idx.get(tokens[2]))
                    {
                        dist[from * n_nodes + to] = cost;
                    }
                }
                "return-to-depot" if tokens.len() >= 2 => {
                    if let Some(&from) = place_to_idx.get(tokens[1]) {
                        dist[from * n_nodes + depot_idx] = cost;
                    }
                }
                "visit-via-depot" if tokens.len() >= 3 => {
                    if let (Some(&from), Some(&to)) =
                        (place_to_idx.get(tokens[1]), place_to_idx.get(tokens[2]))
                    {
                        via_dist[from * n_nodes + to] = cost;
                    }
                }
                _ => {}
            }
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if dist[i * n_nodes + j] >= INF / 2.0
                    && dist[j * n_nodes + i] < INF / 2.0
                {
                    dist[i * n_nodes + j] = dist[j * n_nodes + i];
                }
                if via_dist[i * n_nodes + j] >= INF / 2.0
                    && via_dist[j * n_nodes + i] < INF / 2.0
                {
                    via_dist[i * n_nodes + j] = via_dist[j * n_nodes + i];
                }
            }
        }

        for from in 1..=n_customers {
            for to in 1..=n_customers {
                let idx = from * n_nodes + to;
                if via_dist[idx] >= INF / 2.0 {
                    let a = dist[from * n_nodes + depot_idx];
                    let b = dist[depot_idx * n_nodes + to];
                    if a < INF / 2.0 && b < INF / 2.0 {
                        via_dist[idx] = a + b;
                    }
                }
            }
        }

        let mut neighbors = Vec::with_capacity(n_nodes);
        for from in 0..n_nodes {
            let mut list: Vec<usize> = (1..=n_customers).filter(|&j| j != from).collect();
            list.sort_by(|&a, &b| {
                let sa = dist[from * n_nodes + a] + dist[a * n_nodes + depot_idx];
                let sb = dist[from * n_nodes + b] + dist[b * n_nodes + depot_idx];
                sa.partial_cmp(&sb)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| dist[from * n_nodes + a].partial_cmp(&dist[from * n_nodes + b]).unwrap_or(Ordering::Equal))
                    .then_with(|| a.cmp(&b))
            });
            neighbors.push(list);
        }

        Ok(Self {
            n_customers,
            n_nodes,
            depot_idx,
            visited_facts,
            loc_var_id,
            loc_value_to_place,
            loc_fallback,
            dist,
            via_dist,
            neighbors,
            demands,
            capacity,
            max_vehicles,
            load_var,
            vehicles_var,
        })
    }

    #[inline]
    fn dist_flat(&self, from: usize, to: usize) -> f64 {
        self.dist[from * self.n_nodes + to]
    }

    #[inline]
    fn route_switch_cost(&self, from: usize, to: usize) -> f64 {
        let direct_via = self.via_dist[from * self.n_nodes + to];
        if direct_via < INF / 2.0 {
            return direct_via;
        }
        let a = self.dist_flat(from, self.depot_idx);
        let b = self.dist_flat(self.depot_idx, to);
        if a < INF / 2.0 && b < INF / 2.0 {
            a + b
        } else {
            INF
        }
    }

    #[inline]
    fn first_unvisited(&self, from: usize, visited: &[bool]) -> Option<usize> {
        self.neighbors[from]
            .iter()
            .copied()
            .find(|&c| !visited[c])
    }

    #[inline]
    fn first_feasible(&self, from: usize, visited: &[bool], max_demand: f64) -> Option<usize> {
        self.neighbors[from]
            .iter()
            .copied()
            .find(|&c| !visited[c] && self.demands[c] <= max_demand + EPS)
    }
}

impl Heuristic for LlmHeuristic {
    fn compute_heuristic(&self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        if eval_state.is_goal() {
            return Ok(0.0);
        }

        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();

        let mut numeric_values = vec![0.0; task.numeric_variables().len()];
        registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
            EvaluationError::ComputationFailed(format!("{err:?}"))
        })?;

        let load = if self.load_var == usize::MAX {
            0.0
        } else {
            numeric_values[self.load_var]
        };
        let vehicles = if self.vehicles_var == usize::MAX {
            1.0
        } else {
            numeric_values[self.vehicles_var]
        };

        let mut propositional_values = vec![0usize; task.variables().len()];
        state.fill_state(registry, &mut propositional_values);

        let current = if let Some(var_id) = self.loc_var_id {
            let value = propositional_values[var_id];
            if value < self.loc_value_to_place.len() {
                self.loc_value_to_place[value].unwrap_or(self.depot_idx)
            } else {
                self.depot_idx
            }
        } else {
            self.loc_fallback
                .iter()
                .find(|(var_id, value, _)| propositional_values[*var_id] == *value)
                .map(|(_, _, idx)| *idx)
                .unwrap_or(self.depot_idx)
        };

        let mut visited = vec![false; self.n_nodes];
        let mut unvisited_count = 0usize;
        for cust_idx in 1..=self.n_customers {
            let (var_id, true_value) = self.visited_facts[cust_idx];
            let is_visited = var_id != usize::MAX
                && propositional_values[var_id] == true_value;
            if is_visited {
                visited[cust_idx] = true;
            } else {
                unvisited_count += 1;
            }
        }

        if unvisited_count == 0 {
            if current == self.depot_idx {
                return Ok(0.0);
            } else {
                return Ok(self.dist_flat(current, self.depot_idx));
            }
        }

        let mut cur = current;
        let mut load = if cur == self.depot_idx { 0.0 } else { load };
        let mut veh = vehicles;
        let mut remaining = unvisited_count;
        let mut h = 0.0;

        while remaining > 0 {
            if cur == self.depot_idx {
                let max_demand = self.capacity - load;
                if let Some(c) = self.first_feasible(self.depot_idx, &visited, max_demand) {
                    h += self.dist_flat(self.depot_idx, c);
                    load += self.demands[c];
                    visited[c] = true;
                    cur = c;
                    remaining -= 1;
                    continue;
                }

                let c = self.first_unvisited(self.depot_idx, &visited).unwrap();
                h += self.dist_flat(self.depot_idx, c) + 1_000_000.0;
                load += self.demands[c];
                visited[c] = true;
                cur = c;
                remaining -= 1;
                continue;
            }

            let max_demand = self.capacity - load;
            if let Some(c) = self.first_feasible(cur, &visited, max_demand) {
                h += self.dist_flat(cur, c);
                load += self.demands[c];
                visited[c] = true;
                cur = c;
                remaining -= 1;
                continue;
            }

            let can_new_route = if self.max_vehicles.is_finite() {
                veh + 1.0 <= self.max_vehicles + EPS
            } else {
                true
            };

            if can_new_route {
                if let Some(c) = self.first_unvisited(self.depot_idx, &visited) {
                    h += self.route_switch_cost(cur, c);
                    load = self.demands[c];
                    veh += 1.0;
                    visited[c] = true;
                    cur = c;
                    remaining -= 1;
                    continue;
                }
            }

            if let Some(c) = self.first_unvisited(cur, &visited) {
                h += self.dist_flat(cur, c) + 1_000_000.0;
                load += self.demands[c];
                visited[c] = true;
                cur = c;
                remaining -= 1;
                continue;
            }

            break;
        }

        h += self.dist_flat(cur, self.depot_idx);

        if !h.is_finite() || h < 0.0 {
            h = 1e12;
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
