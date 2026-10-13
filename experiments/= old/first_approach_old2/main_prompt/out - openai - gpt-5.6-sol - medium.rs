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


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

#[derive(Clone, Copy)]
struct FactSlot {
    variable: usize,
    value: usize,
}

#[derive(Clone, Copy)]
struct SavingsEdge {
    a: usize,
    b: usize,
    saving: f64,
}

struct LlmHeuristic {
    customer_names: Vec<String>,
    visited_slots: Vec<FactSlot>,
    location_slots: Vec<FactSlot>,
    depot_location: FactSlot,
    depot_index: usize,
    distances: Vec<f64>,
    savings_edges: Vec<SavingsEdge>,
    vehicles_id: Option<usize>,
    max_vehicles: Option<f64>,
}

impl LlmHeuristic {
    fn parse_positive_atom<'a>(name: &'a str, predicate: &str) -> Option<&'a str> {
        let prefix = format!("Atom {}(", predicate);
        if name.starts_with(&prefix) && name.ends_with(')') {
            let start = prefix.len();
            return Some(&name[start..name.len() - 1]);
        }
        None
    }

    fn distance(&self, from: usize, to: usize) -> f64 {
        let width = self.depot_index + 1;
        self.distances[from * width + to]
    }

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != x {
            let next = parent[x];
            parent[x] = root;
            x = next;
        }
        root
    }

    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut vehicles_id = None;
        let mut max_vehicles = None;
        let initial_numeric = task.get_initial_numeric_state_values();

        for (id, var) in task.numeric_variables().iter().enumerate() {
            let normalized = var.name().to_ascii_lowercase().replace('_', "-");
            if normalized.contains("vehicles()") && !normalized.contains("max-vehicles") {
                vehicles_id = Some(id);
            }
            if normalized.contains("max-vehicles") {
                if let Some(value) = initial_numeric.get(id) {
                    if value.is_finite() && *value >= 1.0 {
                        max_vehicles = Some(*value);
                    }
                }
            }
        }

        let mut visited_by_name: HashMap<String, FactSlot> = HashMap::new();
        let mut location_by_name: HashMap<String, FactSlot> = HashMap::new();

        for (var_id, var) in task.variables().iter().enumerate() {
            for value in 0..var.domain_size() {
                let fact = ExplicitFact::propositional(var_id, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = Self::parse_positive_atom(&fact_name, "visited") {
                    visited_by_name.insert(
                        object.to_string(),
                        FactSlot {
                            variable: var_id,
                            value,
                        },
                    );
                }

                if let Some(object) = Self::parse_positive_atom(&fact_name, "loc") {
                    location_by_name.insert(
                        object.to_string(),
                        FactSlot {
                            variable: var_id,
                            value,
                        },
                    );
                }
            }
        }

        if visited_by_name.is_empty() {
            return Err(
                "llm_001 could not discover any positive visited(customer) facts"
                    .to_string()
                    .into(),
            );
        }

        let mut customer_names: Vec<String> = visited_by_name.keys().cloned().collect();
        customer_names.sort();

        let depot_name = location_by_name
            .keys()
            .find(|name| !visited_by_name.contains_key(*name))
            .cloned()
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not identify the depot location fact"
                    .to_string()
                    .into()
            })?;

        let depot_location = *location_by_name
            .get(&depot_name)
            .ok_or_else(|| -> HeuristicBuildError {
                "llm_001 could not retrieve the depot location slot"
                    .to_string()
                    .into()
            })?;

        let mut customer_index = HashMap::new();
        let mut visited_slots = Vec::with_capacity(customer_names.len());
        let mut location_slots = Vec::with_capacity(customer_names.len());

        for (index, name) in customer_names.iter().enumerate() {
            customer_index.insert(name.clone(), index);
            visited_slots.push(*visited_by_name.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    "llm_001 found an incomplete visited-fact mapping"
                        .to_string()
                        .into()
                },
            )?);
            location_slots.push(*location_by_name.get(name).ok_or_else(
                || -> HeuristicBuildError {
                    "llm_001 found a customer without a loc(customer) fact"
                        .to_string()
                        .into()
                },
            )?);
        }

        let depot_index = customer_names.len();
        let width = depot_index + 1;
        let mut distances = vec![f64::INFINITY; width * width];
        for i in 0..width {
            distances[i * width + i] = 0.0;
        }

        for op in task.get_operators() {
            let words: Vec<&str> = op.name().split_whitespace().collect();
            let cost = metric_operator_cost_from_initial_values(task, op);
            if !cost.is_finite() || cost < 0.0 {
                continue;
            }

            match words.as_slice() {
                ["visit", from, to] => {
                    let to_index = match customer_index.get(*to) {
                        Some(index) => *index,
                        None => continue,
                    };
                    let from_index = if *from == depot_name {
                        depot_index
                    } else if let Some(index) = customer_index.get(*from) {
                        *index
                    } else {
                        continue;
                    };
                    distances[from_index * width + to_index] = cost;
                }
                ["return-to-depot", from] => {
                    if let Some(from_index) = customer_index.get(*from) {
                        distances[*from_index * width + depot_index] = cost;
                    }
                }
                _ => {}
            }
        }

        // The benchmark distances are complete, but use reverse arcs as a robust
        // fallback for symmetric instances if a grounded direct arc was omitted.
        for i in 0..width {
            for j in 0..width {
                if !distances[i * width + j].is_finite()
                    && distances[j * width + i].is_finite()
                {
                    distances[i * width + j] = distances[j * width + i];
                }
            }
        }

        for customer in 0..depot_index {
            if !distances[depot_index * width + customer].is_finite()
                || !distances[customer * width + depot_index].is_finite()
            {
                return Err(
                    "llm_001 could not reconstruct all depot/customer travel costs"
                        .to_string()
                        .into(),
                );
            }
        }

        let mut savings_edges = Vec::new();
        for a in 0..depot_index {
            for b in (a + 1)..depot_index {
                let ab = distances[a * width + b];
                let ba = distances[b * width + a];
                if !ab.is_finite() && !ba.is_finite() {
                    continue;
                }

                let saving_ab = if ab.is_finite() {
                    distances[a * width + depot_index]
                        + distances[depot_index * width + b]
                        - ab
                } else {
                    f64::NEG_INFINITY
                };
                let saving_ba = if ba.is_finite() {
                    distances[b * width + depot_index]
                        + distances[depot_index * width + a]
                        - ba
                } else {
                    f64::NEG_INFINITY
                };

                savings_edges.push(SavingsEdge {
                    a,
                    b,
                    saving: saving_ab.max(saving_ba),
                });
            }
        }

        savings_edges.sort_by(|left, right| {
            right
                .saving
                .partial_cmp(&left.saving)
                .unwrap_or(Ordering::Equal)
        });

        Ok(Self {
            customer_names,
            visited_slots,
            location_slots,
            depot_location,
            depot_index,
            distances,
            savings_edges,
            vehicles_id,
            max_vehicles,
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

        let mut propositional_values = Vec::new();
        state.fill_state(registry, &mut propositional_values);

        let customer_count = self.customer_names.len();
        let mut active = vec![false; customer_count];
        let mut unvisited_count = 0usize;

        for customer in 0..customer_count {
            let slot = self.visited_slots[customer];
            let visited = propositional_values[slot.variable] == slot.value;
            if !visited {
                active[customer] = true;
                unvisited_count += 1;
            }
        }

        let depot_hold = propositional_values[self.depot_location.variable]
            == self.depot_location.value;
        let mut current_customer = None;
        if !depot_hold {
            for customer in 0..customer_count {
                let slot = self.location_slots[customer];
                if propositional_values[slot.variable] == slot.value {
                    current_customer = Some(customer);
                    break;
                }
            }
        }

        if unvisited_count == 0 {
            if let Some(current) = current_customer {
                return Ok(self.distance(current, self.depot_index).max(0.0));
            }
            return Ok(0.0);
        }

        // Start from the cost of serving every remaining customer on its own
        // depot-to-customer-to-depot route. If a route is already in progress,
        // its baseline contribution is only the return from the current point.
        let mut estimate = 0.0;
        for customer in 0..customer_count {
            if active[customer] {
                estimate += self.distance(self.depot_index, customer)
                    + self.distance(customer, self.depot_index);
            }
        }

        let mut active_components = unvisited_count;
        if let Some(current) = current_customer {
            active[current] = true;
            active_components += 1;
            estimate += self.distance(current, self.depot_index);
        }

        let mut available_routes = 1usize;
        if let (Some(vehicles_id), Some(max_vehicles)) =
            (self.vehicles_id, self.max_vehicles)
        {
            let mut numeric_values = Vec::new();
            registry
                .view(state)
                .fill_numeric(&mut numeric_values);
            if let Some(vehicles) = numeric_values.get(vehicles_id) {
                let used = vehicles.round().max(1.0);
                available_routes = (max_vehicles.round() - used)
                    .max(0.0) as usize
                    + 1;
            }
        }

        let target_components = available_routes
            .max(1)
            .min(active_components);
        let links_needed = active_components.saturating_sub(target_components);

        let mut parent: Vec<usize> = (0..customer_count).collect();
        let mut degree = vec![0u8; customer_count];
        let mut links_selected = 0usize;

        for edge in &self.savings_edges {
            if links_selected >= links_needed {
                break;
            }
            if !active[edge.a] || !active[edge.b] {
                continue;
            }

            let limit_a = if current_customer == Some(edge.a) { 1 } else { 2 };
            let limit_b = if current_customer == Some(edge.b) { 1 } else { 2 };
            if degree[edge.a] >= limit_a || degree[edge.b] >= limit_b {
                continue;
            }

            let root_a = Self::find(&mut parent, edge.a);
            let root_b = Self::find(&mut parent, edge.b);
            if root_a == root_b {
                continue;
            }

            let saving = if current_customer == Some(edge.a) {
                let direct = self.distance(edge.a, edge.b);
                if direct.is_finite() {
                    self.distance(edge.a, self.depot_index)
                        + self.distance(self.depot_index, edge.b)
                        - direct
                } else {
                    continue;
                }
            } else if current_customer == Some(edge.b) {
                let direct = self.distance(edge.b, edge.a);
                if direct.is_finite() {
                    self.distance(edge.b, self.depot_index)
                        + self.distance(self.depot_index, edge.a)
                        - direct
                } else {
                    continue;
                }
            } else {
                edge.saving
            };

            parent[root_b] = root_a;
            degree[edge.a] += 1;
            degree[edge.b] += 1;
            estimate -= saving;
            links_selected += 1;
        }

        Ok(estimate.max(0.0))
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
        },
    ])
}
