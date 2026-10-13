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


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

fn parse_atom_object(fact_name: &str, predicate: &str) -> Option<String> {
    let atom = fact_name.trim().strip_prefix("Atom ")?.trim();
    let prefix = format!("{}(", predicate);
    let object = atom.strip_prefix(&prefix)?.strip_suffix(')')?.trim();

    if object.is_empty() || object.contains(',') {
        None
    } else {
        Some(object.to_string())
    }
}

#[derive(Clone, Copy)]
struct Proposition {
    variable: usize,
    value: usize,
}

struct LlmHeuristic {
    customer_count: usize,
    depot_index: usize,
    visited_facts: Vec<Proposition>,
    location_facts: Vec<Option<Proposition>>,
    direct_cost: Vec<f64>,
    via_depot_cost: Vec<f64>,
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        // First recover the customer objects and the proposition encoding from
        // semantic fact names. SAS variable/value IDs are deliberately not
        // assumed to have any particular meaning.
        let mut visited_entries: Vec<(String, Proposition)> = Vec::new();
        let mut location_entries: Vec<(String, Proposition)> = Vec::new();

        for variable in 0..task.variables().len() {
            let domain_size = task.variables()[variable].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(variable, value);
                let fact_name = task.get_fact_name(&fact);

                if let Some(object) = parse_atom_object(fact_name, "visited") {
                    visited_entries.push((
                        object,
                        Proposition { variable, value },
                    ));
                }

                if let Some(object) = parse_atom_object(fact_name, "loc") {
                    location_entries.push((
                        object,
                        Proposition { variable, value },
                    ));
                }
            }
        }

        visited_entries.sort_by(|a, b| a.0.cmp(&b.0));
        visited_entries.dedup_by(|a, b| a.0 == b.0);

        let customer_count = visited_entries.len();
        let depot_index = customer_count;
        let node_count = customer_count + 1;

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        let mut visited_facts = Vec::with_capacity(customer_count);

        for (index, (name, proposition)) in visited_entries.iter().enumerate() {
            node_by_name.insert(name.clone(), index);
            visited_facts.push(*proposition);
        }

        // In this domain the unique location object that is not a customer is
        // the depot. This avoids hardcoding the depot object's PDDL name.
        let depot_name = location_entries
            .iter()
            .find(|(name, _)| !node_by_name.contains_key(name))
            .map(|(name, _)| name.clone());

        if let Some(name) = depot_name.as_ref() {
            node_by_name.insert(name.clone(), depot_index);
        }

        let mut location_facts = vec![None; node_count];
        for (name, proposition) in location_entries {
            if let Some(&index) = node_by_name.get(&name) {
                location_facts[index] = Some(proposition);
            }
        }

        let mut direct_cost = vec![f64::INFINITY; node_count * node_count];
        let mut via_depot_cost = vec![f64::INFINITY; node_count * node_count];

        for index in 0..node_count {
            direct_cost[index * node_count + index] = 0.0;
        }

        // Reconstruct the routing graph from grounded action names and their
        // evaluated metric costs. This remains available even when static
        // travel-cost fluents have been compiled into opaque derived PNEs.
        for operator in task.get_operators() {
            let clean_name = operator
                .name()
                .trim()
                .trim_start_matches('(')
                .trim_end_matches(')');
            let parts: Vec<&str> = clean_name.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let cost = metric_operator_cost_from_initial_values(task, operator);
            if !cost.is_finite() {
                continue;
            }
            let cost = cost.max(0.0);

            match parts[0] {
                "visit" if parts.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(parts[1]), node_by_name.get(parts[2]))
                    {
                        let slot = from * node_count + to;
                        direct_cost[slot] = direct_cost[slot].min(cost);
                    }
                }
                "visit-via-depot" if parts.len() == 3 => {
                    if let (Some(&from), Some(&to)) =
                        (node_by_name.get(parts[1]), node_by_name.get(parts[2]))
                    {
                        let slot = from * node_count + to;
                        via_depot_cost[slot] = via_depot_cost[slot].min(cost);
                    }
                }
                "return-to-depot" if parts.len() == 2 => {
                    if let Some(&from) = node_by_name.get(parts[1]) {
                        let slot = from * node_count + depot_index;
                        direct_cost[slot] = direct_cost[slot].min(cost);
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            customer_count,
            depot_index,
            visited_facts,
            location_facts,
            direct_cost,
            via_depot_cost,
        })
    }

    fn node_count(&self) -> usize {
        self.customer_count + 1
    }

    fn directed_cost(&self, from: usize, to: usize) -> f64 {
        if from == to {
            return 0.0;
        }

        let node_count = self.node_count();
        let slot = from * node_count + to;
        let mut cost = self.direct_cost[slot];

        // A via-depot action is an alternative transition only between two
        // customers. Usually the direct edge is cheaper, but retaining this
        // fallback also handles incomplete compiled direct graphs.
        if from < self.customer_count && to < self.customer_count {
            cost = cost.min(self.via_depot_cost[slot]);
        }

        cost
    }

    fn undirected_cost(&self, a: usize, b: usize) -> f64 {
        self.directed_cost(a, b).min(self.directed_cost(b, a))
    }

    fn mst_cost(&self, vertices: &[usize]) -> f64 {
        if vertices.len() <= 1 {
            return 0.0;
        }

        let count = vertices.len();
        let mut in_tree = vec![false; count];
        let mut best = vec![f64::INFINITY; count];
        best[0] = 0.0;
        let mut total = 0.0;

        for _ in 0..count {
            let mut selected = None;
            let mut selected_cost = f64::INFINITY;

            for index in 0..count {
                if !in_tree[index] && best[index] < selected_cost {
                    selected = Some(index);
                    selected_cost = best[index];
                }
            }

            let Some(selected) = selected else {
                break;
            };

            in_tree[selected] = true;
            total += selected_cost.max(0.0);

            for other in 0..count {
                if !in_tree[other] {
                    let edge = self.undirected_cost(
                        vertices[selected],
                        vertices[other],
                    );
                    if edge < best[other] {
                        best[other] = edge;
                    }
                }
            }
        }

        total
    }

    fn greedy_completion_cost(
        &self,
        start: usize,
        unvisited: &[usize],
    ) -> f64 {
        let mut remaining = unvisited.to_vec();
        let mut current = start;
        let mut total = 0.0;

        while !remaining.is_empty() {
            let mut best_position = None;
            let mut best_cost = f64::INFINITY;

            for (position, &customer) in remaining.iter().enumerate() {
                let cost = self.directed_cost(current, customer);
                if cost < best_cost {
                    best_cost = cost;
                    best_position = Some(position);
                }
            }

            let Some(position) = best_position else {
                break;
            };

            if !best_cost.is_finite() {
                break;
            }

            total += best_cost.max(0.0);
            current = remaining.swap_remove(position);
        }

        let return_cost = self.directed_cost(current, self.depot_index);
        if return_cost.is_finite() {
            total += return_cost.max(0.0);
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

        let mut unvisited = Vec::with_capacity(self.customer_count);
        for customer in 0..self.customer_count {
            let proposition = self.visited_facts[customer];
            let fact = ExplicitFact::propositional(
                proposition.variable,
                proposition.value,
            );
            if !fact.is_hold(registry.view(state)) {
                unvisited.push(customer);
            }
        }

        let mut current = self.depot_index;
        for node in 0..self.location_facts.len() {
            if let Some(proposition) = self.location_facts[node] {
                let fact = ExplicitFact::propositional(
                    proposition.variable,
                    proposition.value,
                );
                if fact.is_hold(registry.view(state)) {
                    current = node;
                    break;
                }
            }
        }

        // Once every customer is served, the remaining routing cost is exactly
        // the final edge back to the depot.
        if unvisited.is_empty() {
            let return_cost = self.directed_cost(current, self.depot_index);
            return Ok(if return_cost.is_finite() {
                return_cost.max(0.0)
            } else {
                0.0
            });
        }

        let greedy = self.greedy_completion_cost(current, &unvisited);

        let mut mst_vertices = Vec::with_capacity(unvisited.len() + 2);
        mst_vertices.push(current);
        for &customer in &unvisited {
            if !mst_vertices.contains(&customer) {
                mst_vertices.push(customer);
            }
        }
        if !mst_vertices.contains(&self.depot_index) {
            mst_vertices.push(self.depot_index);
        }

        let mst = self.mst_cost(&mst_vertices);

        // Greedy routing gives a concrete, strongly discriminating completion
        // estimate, while the MST component makes the estimate less sensitive
        // to individual nearest-neighbor choices.
        let h = 0.8 * greedy + 0.2 * mst;

        Ok(if h.is_finite() { h.max(0.0) } else { 0.0 })
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
