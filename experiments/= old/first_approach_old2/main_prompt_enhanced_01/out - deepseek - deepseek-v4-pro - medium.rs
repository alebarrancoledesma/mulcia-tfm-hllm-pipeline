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

use std::collections::HashMap;
use std::cell::RefCell;

const LARGE_PENALTY: f64 = 1e12;
const INF: f64 = 1e18;

// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

struct LlmHeuristic {
    num_places: usize,
    num_customers: usize,
    depot_idx: usize,

    loc_facts: Vec<ExplicitFact>,      // indexed by place_idx
    visited_facts: Vec<ExplicitFact>,  // indexed by customer_idx (0..num_customers)

    load_id: usize,
    vehicles_id: usize,
    capacity: f64,
    max_vehicles: f64,

    demand: Vec<f64>,                   // indexed by customer_idx

    direct_cost: Vec<Vec<f64>>,        // [place_idx][place_idx]
    via_depot_cost: Vec<Vec<f64>>,    // [place_idx][place_idx]

    nearest_direct: Vec<Vec<usize>>,   // for each place, list of customer indices sorted by direct cost
    nearest_via: Vec<Vec<usize>>,      // for each customer place, list of customer indices sorted by via-depot cost

    scratch_numeric: RefCell<Vec<f64>>,
    scratch_visited: RefCell<Vec<bool>>,
}

impl LlmHeuristic {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, HeuristicBuildError> {
        // --------------------------------------------------------------------
        // 1. Scan propositional facts to discover loc and visited
        // --------------------------------------------------------------------
        let mut loc_facts_map: HashMap<String, ExplicitFact> = HashMap::new();
        let mut visited_facts_map: HashMap<String, ExplicitFact> = HashMap::new();

        for var_id in 0..task.variables().len() {
            let var = &task.variables()[var_id];
            let domain_size = var.domain_size();
            for val in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, val);
                let name = task.get_fact_name(&fact).to_string();

                if let Some(rest) = name.strip_prefix("Atom loc(") {
                    if let Some(close) = rest.find(')') {
                        let place_name = rest[..close].to_string();
                        loc_facts_map.insert(place_name, fact);
                    }
                } else if let Some(rest) = name.strip_prefix("Atom visited(") {
                    if let Some(close) = rest.find(')') {
                        let cust_name = rest[..close].to_string();
                        visited_facts_map.insert(cust_name, fact);
                    }
                }
            }
        }

        if loc_facts_map.is_empty() || visited_facts_map.is_empty() {
            return Err("Could not find loc or visited propositional facts".to_string().into());
        }

        // Identify depot: a place that appears in loc facts but not in visited facts.
        let customer_names: Vec<String> = visited_facts_map.keys().cloned().collect();
        let mut depot_name: Option<String> = None;
        for place_name in loc_facts_map.keys() {
            if !customer_names.contains(place_name) {
                depot_name = Some(place_name.clone());
                break;
            }
        }
        let depot_name = depot_name.ok_or_else(|| HeuristicBuildError::from("Could not identify depot".to_string()))?;

        // Build place index: depot = 0, customers = 1..
        let mut place_names: Vec<String> = Vec::new();
        place_names.push(depot_name.clone());
        let mut sorted_customers: Vec<String> = customer_names.clone();
        sorted_customers.sort();
        place_names.extend(sorted_customers.clone());

        let num_places = place_names.len();
        let num_customers = sorted_customers.len();
        let depot_idx = 0;

        let mut name_to_place_idx: HashMap<String, usize> = HashMap::new();
        for (idx, name) in place_names.iter().enumerate() {
            name_to_place_idx.insert(name.clone(), idx);
        }

        // Build vectors of facts aligned to indices
        let mut loc_facts: Vec<ExplicitFact> = Vec::with_capacity(num_places);
        for name in &place_names {
            loc_facts.push(loc_facts_map[name].clone());
        }
        let mut visited_facts: Vec<ExplicitFact> = Vec::with_capacity(num_customers);
        for name in &sorted_customers {
            visited_facts.push(visited_facts_map[name].clone());
        }

        // --------------------------------------------------------------------
        // 2. Discover numeric variable IDs for load, vehicles, capacity, max_vehicles
        // --------------------------------------------------------------------
        let mut load_id: Option<usize> = None;
        let mut vehicles_id: Option<usize> = None;
        let mut capacity_id: Option<usize> = None;
        let mut max_vehicles_id: Option<usize> = None;
        let mut demand_var_ids: HashMap<String, usize> = HashMap::new(); // customer name -> numeric var id

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if name == "load()" || name == "PNE load()" {
                load_id = Some(id);
            } else if name == "vehicles()" || name == "PNE vehicles()" {
                vehicles_id = Some(id);
            } else if name == "capacity()" || name == "PNE capacity()" {
                capacity_id = Some(id);
            } else if name == "max_vehicles()" || name == "PNE max_vehicles()" {
                max_vehicles_id = Some(id);
            } else if let Some(start) = name.find("demand(") {
                if let Some(inner) = extract_inner(&name, start + 7) {
                    demand_var_ids.insert(inner.to_string(), id);
                }
            }
        }

        let load_id = load_id.ok_or_else(|| HeuristicBuildError::from("Missing load() numeric variable".to_string()))?;
        let vehicles_id = vehicles_id.ok_or_else(|| HeuristicBuildError::from("Missing vehicles() numeric variable".to_string()))?;

        // Read initial numeric values
        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_id.map(|id| initial_numeric[id]).unwrap_or(100.0); // fallback if missing
        let max_vehicles = max_vehicles_id.map(|id| initial_numeric[id]).unwrap_or(5.0);

        // Demand for each customer
        let mut demand: Vec<f64> = Vec::with_capacity(num_customers);
        for cust_name in &sorted_customers {
            let val = demand_var_ids
                .get(cust_name)
                .map(|&id| initial_numeric[id])
                .unwrap_or(1.0); // fallback: assume 1 if not found
            demand.push(val);
        }

        // --------------------------------------------------------------------
        // 3. Extract travel costs from grounded operators
        // --------------------------------------------------------------------
        let mut direct_cost: Vec<Vec<f64>> = vec![vec![INF; num_places]; num_places];
        let mut via_depot_cost: Vec<Vec<f64>> = vec![vec![INF; num_places]; num_places];

        for op in task.get_operators() {
            let name = op.name().to_string();
            let parts: Vec<&str> = name.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let cost = metric_operator_cost_from_initial_values(task, op);
            match parts[0] {
                "visit" if parts.len() == 3 => {
                    let from = parts[1];
                    let to = parts[2];
                    if let (Some(&from_idx), Some(&to_idx)) = (name_to_place_idx.get(from), name_to_place_idx.get(to)) {
                        direct_cost[from_idx][to_idx] = cost;
                    }
                }
                "visit-via-depot" if parts.len() == 3 => {
                    let from = parts[1];
                    let to = parts[2];
                    if let (Some(&from_idx), Some(&to_idx)) = (name_to_place_idx.get(from), name_to_place_idx.get(to)) {
                        via_depot_cost[from_idx][to_idx] = cost;
                    }
                }
                "return-to-depot" if parts.len() == 2 => {
                    let from = parts[1];
                    if let Some(&from_idx) = name_to_place_idx.get(from) {
                        direct_cost[from_idx][depot_idx] = cost;
                    }
                }
                _ => {}
            }
        }

        // --------------------------------------------------------------------
        // 4. Fallback: try to fill missing costs from travel-cost numeric vars
        // --------------------------------------------------------------------
        for (id, var) in task.numeric_variables().iter().enumerate() {
            let name = var.name().to_string();
            if let Some(start) = name.find("travel-cost(") {
                if let Some(inner) = extract_inner(&name, start + 12) {
                    let mut args = inner.split(',');
                    if let (Some(a), Some(b)) = (args.next(), args.next()) {
                        let a = a.trim();
                        let b = b.trim();
                        if let (Some(&a_idx), Some(&b_idx)) = (name_to_place_idx.get(a), name_to_place_idx.get(b)) {
                            direct_cost[a_idx][b_idx] = initial_numeric[id];
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // 5. Fill any remaining missing arcs conservatively
        // --------------------------------------------------------------------
        // Ensure depot distances are known if possible
        for i in 0..num_places {
            if i != depot_idx {
                if direct_cost[depot_idx][i] >= INF && !via_depot_cost[depot_idx][i].is_finite() {
                    // try to infer from reverse? maybe via depot sum after both legs known
                }
                if direct_cost[i][depot_idx] >= INF {
                    // try via depot: cost i->depot = via_depot_cost[i][?] not directly
                }
            }
        }
        // For any missing direct_cost[i][j] where i,j are customers, use via_depot if available
        for i in 1..num_places {
            for j in 1..num_places {
                if i != j && direct_cost[i][j] >= INF {
                    if via_depot_cost[i][j] < INF {
                        direct_cost[i][j] = via_depot_cost[i][j];
                    } else {
                        // last resort: via depot using two legs if known
                        let leg1 = direct_cost[i][depot_idx];
                        let leg2 = direct_cost[depot_idx][j];
                        if leg1 < INF && leg2 < INF {
                            direct_cost[i][j] = leg1 + leg2;
                            via_depot_cost[i][j] = leg1 + leg2;
                        }
                    }
                }
                if via_depot_cost[i][j] >= INF {
                    let leg1 = direct_cost[i][depot_idx];
                    let leg2 = direct_cost[depot_idx][j];
                    if leg1 < INF && leg2 < INF {
                        via_depot_cost[i][j] = leg1 + leg2;
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // 6. Precompute sorted neighbour lists
        // --------------------------------------------------------------------
        let mut nearest_direct: Vec<Vec<usize>> = Vec::with_capacity(num_places);
        let mut nearest_via: Vec<Vec<usize>> = Vec::with_capacity(num_places);

        for p_idx in 0..num_places {
            let mut direct_neighbors: Vec<usize> = (1..num_places).collect(); // customer place indices
            direct_neighbors.sort_by(|&a, &b| {
                direct_cost[p_idx][a]
                    .partial_cmp(&direct_cost[p_idx][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // convert to customer indices (place_idx - 1)
            let direct_cust_indices: Vec<usize> = direct_neighbors.iter().map(|&p| p - 1).collect();
            nearest_direct.push(direct_cust_indices);

            let via_cust_indices: Vec<usize>;
            if p_idx == depot_idx {
                // not used; fill with empty or same as direct
                via_cust_indices = nearest_direct[depot_idx].clone();
            } else {
                let mut via_neighbors: Vec<usize> = (1..num_places).collect();
                via_neighbors.sort_by(|&a, &b| {
                    via_depot_cost[p_idx][a]
                        .partial_cmp(&via_depot_cost[p_idx][b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                via_cust_indices = via_neighbors.iter().map(|&p| p - 1).collect();
            }
            nearest_via.push(via_cust_indices);
        }

        Ok(Self {
            num_places,
            num_customers,
            depot_idx,
            loc_facts,
            visited_facts,
            load_id,
            vehicles_id,
            capacity,
            max_vehicles,
            demand,
            direct_cost,
            via_depot_cost,
            nearest_direct,
            nearest_via,
            scratch_numeric: RefCell::new(vec![0.0; task.numeric_variables().len()]),
            scratch_visited: RefCell::new(vec![false; num_customers]),
        })
    }
}

fn extract_inner(s: &str, start: usize) -> Option<&str> {
    let open = s[start..].find('(')?;
    let close = s[start + open + 1..].find(')')?;
    Some(&s[start + open + 1..start + open + 1 + close])
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

        // Read numeric state
        {
            let mut numeric = self.scratch_numeric.borrow_mut();
            registry.view(state).fill_numeric(&mut numeric);
            let load_val = numeric[self.load_id];
            let vehicles_val = numeric[self.vehicles_id];
            // capacity and max_vehicles are static, stored in self
        }
        let load_val = {
            let numeric = self.scratch_numeric.borrow();
            numeric[self.load_id]
        };
        let vehicles_val = {
            let numeric = self.scratch_numeric.borrow();
            numeric[self.vehicles_id]
        };

        // Determine current location
        let view = registry.view(state);
        let mut cur_idx = self.depot_idx;
        for idx in 0..self.num_places {
            if self.loc_facts[idx].is_hold(view) {
                cur_idx = idx;
                break;
            }
        }

        // Build visited mask for customers
        let mut visited = self.scratch_visited.borrow_mut();
        visited.clear();
        visited.resize(self.num_customers, false);
        for c in 0..self.num_customers {
            visited[c] = self.visited_facts[c].is_hold(view);
        }

        // Compute remaining demand and unvisited count
        let mut remaining_demand = 0.0;
        let mut unvisited_count = 0;
        for c in 0..self.num_customers {
            if !visited[c] {
                remaining_demand += self.demand[c];
                unvisited_count += 1;
            }
        }

        // Resource infeasibility check
        let available_capacity = (self.max_vehicles - vehicles_val).max(0.0) * self.capacity
            + (self.capacity - load_val).max(0.0);
        if remaining_demand > available_capacity + 1e-6 {
            return Ok(LARGE_PENALTY);
        }

        // If all visited, only need to return to depot
        if unvisited_count == 0 {
            if cur_idx == self.depot_idx {
                return Ok(0.0);
            } else {
                let cost = self.direct_cost[cur_idx][self.depot_idx];
                if cost >= INF {
                    return Ok(LARGE_PENALTY);
                }
                return Ok(cost);
            }
        }

        // Greedy simulation of remaining route
        let mut total_est = 0.0;
        let mut sim_load = load_val;
        let mut sim_vehicles = vehicles_val;
        let mut sim_cur = cur_idx;
        let mut remaining = unvisited_count;

        while remaining > 0 {
            let mut best_direct: Option<(usize, f64)> = None;
            let mut best_via: Option<(usize, f64)> = None;

            // Direct visits
            let neighbor_list = if sim_cur == self.depot_idx {
                &self.nearest_direct[self.depot_idx]
            } else {
                &self.nearest_direct[sim_cur]
            };
            for &cust_idx in neighbor_list {
                if !visited[cust_idx] {
                    let to_place = cust_idx + 1;
                    let cost = self.direct_cost[sim_cur][to_place];
                    if cost < INF && sim_load + self.demand[cust_idx] <= self.capacity + 1e-6 {
                        best_direct = Some((cust_idx, cost));
                        break;
                    }
                }
            }

            // Via-depot visits (only from a customer, not from depot)
            if sim_cur != self.depot_idx && sim_vehicles < self.max_vehicles {
                let via_list = &self.nearest_via[sim_cur];
                for &cust_idx in via_list {
                    if !visited[cust_idx] {
                        let to_place = cust_idx + 1;
                        let cost = self.via_depot_cost[sim_cur][to_place];
                        if cost < INF && self.demand[cust_idx] <= self.capacity + 1e-6 {
                            best_via = Some((cust_idx, cost));
                            break;
                        }
                    }
                }
            }

            // Choose the best feasible transition
            let chosen = match (best_direct, best_via) {
                (Some(d), Some(v)) => {
                    if d.1 <= v.1 {
                        (d.0, d.1, false)
                    } else {
                        (v.0, v.1, true)
                    }
                }
                (Some(d), None) => (d.0, d.1, false),
                (None, Some(v)) => (v.0, v.1, true),
                (None, None) => {
                    return Ok(LARGE_PENALTY);
                }
            };

            let (cust_idx, cost, via) = chosen;
            total_est += cost;

            if via {
                sim_load = self.demand[cust_idx];
                sim_vehicles += 1.0;
            } else {
                sim_load += self.demand[cust_idx];
            }
            sim_cur = cust_idx + 1;
            visited[cust_idx] = true;
            remaining -= 1;
        }

        // Final return to depot
        if sim_cur != self.depot_idx {
            let ret_cost = self.direct_cost[sim_cur][self.depot_idx];
            if ret_cost >= INF {
                return Ok(LARGE_PENALTY);
            }
            total_est += ret_cost;
        }

        Ok(total_est)
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
