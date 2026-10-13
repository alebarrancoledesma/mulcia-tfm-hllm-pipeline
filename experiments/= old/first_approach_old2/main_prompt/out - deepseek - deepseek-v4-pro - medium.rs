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

use std::collections::{HashMap, HashSet};

// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

fn parse_inside(name: &str, prefix: &str) -> Option<String> {
    let start = name.find(prefix)? + prefix.len();
    let after = &name[start..];
    let end = after.find(')')?;
    Some(after[..end].to_string())
}

fn parse_visited(name: &str) -> Option<String> {
    parse_inside(name, "visited(")
}

fn parse_loc(name: &str) -> Option<String> {
    parse_inside(name, "loc(")
}

fn parse_visit(name: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = name.split_whitespace().collect();
    if parts.len() == 3 && parts[0] == "visit" {
        Some((parts[1].to_string(), parts[2].to_string()))
    } else {
        None
    }
}

fn parse_visit_via_depot(name: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = name.split_whitespace().collect();
    if parts.len() == 3 && parts[0] == "visit-via-depot" {
        Some((parts[1].to_string(), parts[2].to_string()))
    } else {
        None
    }
}

fn parse_return_to_depot(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split_whitespace().collect();
    if parts.len() == 2 && parts[0] == "return-to-depot" {
        Some(parts[1].to_string())
    } else {
        None
    }
}

fn is_exact_or_spaced_suffix(name: &str, marker: &str) -> bool {
    name == marker
        || name.ends_with(&format!(" {}", marker))
        || name.ends_with(marker)
}

fn node_index(
    name: &str,
    depot_name: &str,
    customer_index: &HashMap<String, usize>,
) -> Option<usize> {
    if name == depot_name {
        Some(0)
    } else {
        customer_index.get(name).copied()
    }
}

struct LlmHeuristic {
    num_customers: usize,
    num_nodes: usize,
    num_numeric_vars: usize,
    customer_index: HashMap<String, usize>,
    depot_name: String,
    visited_facts: Vec<ExplicitFact>,
    loc_facts: Vec<ExplicitFact>,
    demands: Vec<f64>,
    capacity: Option<f64>,
    max_vehicles: Option<f64>,
    load_id: Option<usize>,
    vehicles_id: Option<usize>,
    dist: Vec<f64>,
    via_depot: Vec<f64>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut customer_set = HashSet::new();
        let mut loc_places = HashSet::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact).to_string();
                if let Some(c) = parse_visited(&name) {
                    customer_set.insert(c);
                }
                if let Some(p) = parse_loc(&name) {
                    loc_places.insert(p);
                }
            }
        }

        let mut customers: Vec<String> = customer_set.into_iter().collect();
        customers.sort();

        let num_customers = customers.len();
        let num_nodes = num_customers + 1;

        let mut customer_index = HashMap::new();
        for (i, c) in customers.iter().enumerate() {
            customer_index.insert(c.clone(), i + 1);
        }

        let depot_name = loc_places
            .iter()
            .find(|p| !customer_index.contains_key(p.as_str()))
            .cloned()
            .unwrap_or_else(|| "d1".to_string());

        let mut visited_facts: Vec<Option<ExplicitFact>> =
            (0..num_customers).map(|_| None).collect();
        let mut loc_facts: Vec<Option<ExplicitFact>> =
            (0..num_nodes).map(|_| None).collect();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact).to_string();

                if !name.starts_with("Atom ") {
                    continue;
                }

                if let Some(c) = parse_visited(&name) {
                    if let Some(&idx) = customer_index.get(&c) {
                        if visited_facts[idx].is_none() {
                            visited_facts[idx] = Some(fact);
                        }
                    }
                }

                if let Some(p) = parse_loc(&name) {
                    let node = if p == depot_name {
                        Some(0)
                    } else {
                        customer_index.get(&p).copied()
                    };
                    if let Some(node_idx) = node {
                        if loc_facts[node_idx].is_none() {
                            loc_facts[node_idx] = Some(fact);
                        }
                    }
                }
            }
        }

        let visited_facts: Vec<ExplicitFact> = visited_facts
            .into_iter()
            .map(|o| {
                o.ok_or_else(|| "missing visited fact".to_string().into())
            })
            .collect::<Result<_, HeuristicBuildError>>()?;

        let loc_facts: Vec<ExplicitFact> = loc_facts
            .into_iter()
            .map(|o| {
                o.ok_or_else(|| "missing loc fact".to_string().into())
            })
            .collect::<Result<_, HeuristicBuildError>>()?;

        let mut dist = vec![f64::INFINITY; num_nodes * num_nodes];
        let mut via_depot = vec![f64::INFINITY; num_nodes * num_nodes];
        for i in 0..num_nodes {
            dist[i * num_nodes + i] = 0.0;
        }

        for op in task.get_operators() {
            let op_name = op.name();
            let cost = metric_operator_cost_from_initial_values(task, op) as f64;

            if let Some((from, to)) = parse_visit(op_name) {
                if let (Some(fi), Some(ti)) = (
                    node_index(&from, &depot_name, &customer_index),
                    node_index(&to, &depot_name, &customer_index),
                ) {
                    dist[fi * num_nodes + ti] = cost;
                }
            } else if let Some((from, to)) = parse_visit_via_depot(op_name) {
                if let (Some(fi), Some(ti)) = (
                    node_index(&from, &depot_name, &customer_index),
                    node_index(&to, &depot_name, &customer_index),
                ) {
                    via_depot[fi * num_nodes + ti] = cost;
                }
            } else if let Some(from) = parse_return_to_depot(op_name) {
                if let Some(fi) = node_index(&from, &depot_name, &customer_index) {
                    dist[fi * num_nodes + 0] = cost;
                }
            }
        }

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let idx = i * num_nodes + j;
                let opp = j * num_nodes + i;
                if !dist[idx].is_finite() && dist[opp].is_finite() {
                    dist[idx] = dist[opp];
                }
                if i != 0
                    && j != 0
                    && !via_depot[idx].is_finite()
                    && dist[i * num_nodes + 0].is_finite()
                    && dist[0 * num_nodes + j].is_finite()
                {
                    via_depot[idx] =
                        dist[i * num_nodes + 0] + dist[0 * num_nodes + j];
                }
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let num_numeric_vars = task.numeric_variables().len();

        let mut demands = vec![0.0; num_customers];
        let mut capacity = None;
        let mut max_vehicles = None;
        let mut load_id = None;
        let mut vehicles_id = None;

        let mut customers_by_len: Vec<String> = customers.clone();
        customers_by_len.sort_by(|a, b| b.len().cmp(&a.len()));

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            let val = initial_numeric.get(id).copied().unwrap_or(0.0);

            if name.contains("demand") {
                let mut assigned = false;
                if let Some(cust) = parse_inside(&name, "demand(") {
                    if let Some(&idx) = customer_index.get(&cust) {
                        demands[idx] = val;
                        assigned = true;
                    }
                }
                if !assigned {
                    for cust in &customers_by_len {
                        if name.contains(cust.as_str()) {
                            if let Some(&idx) = customer_index.get(cust) {
                                demands[idx] = val;
                                assigned = true;
                                break;
                            }
                        }
                    }
                }
            } else if name.contains("max_vehicles") {
                max_vehicles = Some(val);
            } else if name.contains("capacity") {
                capacity = Some(val);
            } else if name.contains("load()") || name == "load()" {
                load_id = Some(id);
            } else if (name.contains("vehicles()") || name == "vehicles()")
                && !name.contains("max_vehicles")
            {
                vehicles_id = Some(id);
            }
        }

        Ok(Self {
            num_customers,
            num_nodes,
            num_numeric_vars,
            customer_index,
            depot_name,
            visited_facts,
            loc_facts,
            demands,
            capacity,
            max_vehicles,
            load_id,
            vehicles_id,
            dist,
            via_depot,
        })
    }

    fn dist(&self, i: usize, j: usize) -> f64 {
        self.dist[i * self.num_nodes + j]
    }

    fn via(&self, i: usize, j: usize) -> f64 {
        self.via_depot[i * self.num_nodes + j]
    }

    fn choose_next(
        &self,
        cur: usize,
        load: f64,
        vehicles: f64,
        visited: &[bool],
        cap: f64,
        max_v: f64,
    ) -> Option<(usize, f64, f64, f64)> {
        let mut best: Option<(usize, f64, f64, f64)> = None;

        for c_idx in 0..self.num_customers {
            if visited[c_idx] {
                continue;
            }
            let node = c_idx + 1;
            let demand = self.demands[c_idx];

            if cur == 0 {
                let cost = self.dist(0, node);
                if demand <= cap && cost.is_finite() {
                    let better = match best {
                        None => true,
                        Some((_, best_cost, _, _)) => cost < best_cost,
                    };
                    if better {
                        best = Some((node, cost, demand, vehicles));
                    }
                }
            } else {
                let cost_direct = self.dist(cur, node);
                if load + demand <= cap && cost_direct.is_finite() {
                    let better = match best {
                        None => true,
                        Some((_, best_cost, _, _)) => cost_direct < best_cost,
                    };
                    if better {
                        best = Some((node, cost_direct, load + demand, vehicles));
                    }
                }

                let cost_via = self.via(cur, node);
                if vehicles + 1.0 <= max_v && demand <= cap && cost_via.is_finite() {
                    let better = match best {
                        None => true,
                        Some((_, best_cost, _, _)) => cost_via < best_cost,
                    };
                    if better {
                        best = Some((node, cost_via, demand, vehicles + 1.0));
                    }
                }
            }
        }

        if best.is_none() {
            for c_idx in 0..self.num_customers {
                if visited[c_idx] {
                    continue;
                }
                let node = c_idx + 1;
                let demand = self.demands[c_idx];

                let (cost, new_load, new_vehicles) = if cur == 0 {
                    (self.dist(0, node), demand, vehicles)
                } else {
                    let direct = self.dist(cur, node);
                    let via = self.via(cur, node);
                    let use_via = via < direct;
                    if use_via {
                        (via, demand, vehicles + 1.0)
                    } else {
                        (direct, load + demand, vehicles)
                    }
                };

                if cost.is_finite() {
                    let better = match best {
                        None => true,
                        Some((_, best_cost, _, _)) => cost < best_cost,
                    };
                    if better {
                        best = Some((node, cost, new_load, new_vehicles));
                    }
                }
            }
        }

        best
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

        let mut visited = vec![false; self.num_customers];
        let mut unvisited_count = 0;

        for (i, fact) in self.visited_facts.iter().enumerate() {
            if fact.is_hold(registry.view(state)) {
                visited[i] = true;
            } else {
                unvisited_count += 1;
            }
        }

        let mut cur = 0;
        for (i, fact) in self.loc_facts.iter().enumerate() {
            if fact.is_hold(registry.view(state)) {
                cur = i;
                break;
            }
        }

        if unvisited_count == 0 {
            if cur != 0 {
                return Ok(self.dist(cur, 0));
            } else {
                return Ok(0.0);
            }
        }

        let mut numeric_values = vec![0.0; self.num_numeric_vars];
        registry.view(state).fill_numeric(&mut numeric_values);

        let current_load = self.load_id.map(|id| numeric_values[id]).unwrap_or(0.0);
        let current_vehicles = self.vehicles_id.map(|id| numeric_values[id]).unwrap_or(1.0);

        let cap = self.capacity.unwrap_or(f64::INFINITY);
        let max_v = self.max_vehicles.unwrap_or(f64::INFINITY);

        let mut route_node = cur;
        let mut load = current_load;
        let mut vehicles = current_vehicles;
        let mut total = 0.0;
        let mut remaining = unvisited_count;

        while remaining > 0 {
            if let Some((next_node, cost, new_load, new_vehicles)) =
                self.choose_next(route_node, load, vehicles, &visited, cap, max_v)
            {
                total += cost;
                route_node = next_node;
                load = new_load;
                vehicles = new_vehicles;
                visited[next_node - 1] = true;
                remaining -= 1;
            } else {
                break;
            }
        }

        if route_node != 0 {
            let return_cost = self.dist(route_node, 0);
            if return_cost.is_finite() {
                total += return_cost;
            }
        }

        Ok(total)
    }

    fn heuristic_name(&self) -> &str {
        "llm_001"
    }
}

// ============================================================================
// PLANFORGE REGISTRATION
// ============================================================================

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
