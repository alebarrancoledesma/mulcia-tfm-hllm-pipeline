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


// ============================================================================
// LLM-GENERATED HEURISTIC
// ============================================================================

#[derive(Clone, Copy)]
struct StateFact {
    var_id: usize,
    value: usize,
}

#[derive(Clone, Copy)]
struct LocationFact {
    fact: StateFact,
    node: usize,
}

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    remaining: Vec<usize>,
    selected: Vec<bool>,
}

struct LlmHeuristic {
    customer_count: usize,
    depot: usize,
    visited_facts: Vec<StateFact>,
    location_facts: Vec<LocationFact>,
    direct_costs: Vec<f64>,
    via_depot_costs: Vec<f64>,
    return_costs: Vec<f64>,
    demands: Vec<f64>,
    demands_complete: bool,
    capacity: Option<f64>,
    load_var_id: Option<usize>,
    scratch: RefCell<ScratchBuffers>,
}

fn atom_argument<'a>(fact_name: &'a str, predicate: &str) -> Option<&'a str> {
    let trimmed = fact_name.trim();
    let atom = trimmed.strip_prefix("Atom ")?;
    let rest = atom.strip_prefix(predicate)?;
    let rest = rest.strip_prefix('(')?;
    let argument = rest.strip_suffix(')')?.trim();

    if argument.is_empty() || argument.chars().any(char::is_whitespace) {
        None
    } else {
        Some(argument)
    }
}

fn exact_numeric_name(name: &str, fluent: &str) -> bool {
    let trimmed = name.trim();
    trimmed == fluent || trimmed.strip_prefix("PNE ") == Some(fluent)
}

fn usable_cost(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

impl LlmHeuristic {
    fn new(
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, HeuristicBuildError> {
        let mut depot_name: Option<String> = None;

        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
            let goal_name = task.get_fact_name(&goal);
            if let Some(place) = atom_argument(goal_name, "loc") {
                depot_name = Some(place.to_string());
                break;
            }
        }

        let depot_name = depot_name.ok_or_else(|| {
            "llm_001 could not identify the depot from the loc goal"
                .to_string()
                .into()
        })?;

        let mut customer_records: Vec<(String, StateFact)> = Vec::new();
        let mut raw_locations: Vec<(String, StateFact)> = Vec::new();

        for var_id in 0..task.variables().len() {
            let domain_size = task.variables()[var_id].domain_size();
            for value in 0..domain_size {
                let fact = ExplicitFact::propositional(var_id, value);
                let name = task.get_fact_name(&fact);

                if let Some(customer) = atom_argument(name, "visited") {
                    customer_records.push((
                        customer.to_string(),
                        StateFact { var_id, value },
                    ));
                }

                if let Some(place) = atom_argument(name, "loc") {
                    raw_locations.push((
                        place.to_string(),
                        StateFact { var_id, value },
                    ));
                }
            }
        }

        customer_records.sort_by(|a, b| a.0.cmp(&b.0));
        customer_records.dedup_by(|a, b| a.0 == b.0);

        let customer_count = customer_records.len();
        let depot = customer_count;
        let node_count = customer_count + 1;

        let mut node_by_name: HashMap<String, usize> = HashMap::new();
        let mut visited_facts = Vec::with_capacity(customer_count);

        for (customer_id, (name, fact)) in customer_records.iter().enumerate() {
            node_by_name.insert(name.clone(), customer_id);
            visited_facts.push(*fact);
        }
        node_by_name.insert(depot_name.clone(), depot);

        let mut location_facts = Vec::new();
        for (name, fact) in raw_locations {
            if let Some(&node) = node_by_name.get(&name) {
                location_facts.push(LocationFact { fact, node });
            }
        }

        if location_facts.is_empty() {
            return Err(
                "llm_001 could not identify any loc facts for CVRP nodes"
                    .to_string()
                    .into(),
            );
        }

        let mut load_var_id = None;
        let mut capacity_var_id = None;

        for (id, variable) in task.numeric_variables().iter().enumerate() {
            if exact_numeric_name(variable.name(), "load()") {
                load_var_id = Some(id);
            } else if exact_numeric_name(variable.name(), "capacity()") {
                capacity_var_id = Some(id);
            }
        }

        let initial_numeric = task.get_initial_numeric_state_values();
        let capacity = capacity_var_id
            .and_then(|id| initial_numeric.get(id).copied())
            .filter(|value| value.is_finite() && *value > 0.0);

        let mut direct_costs = vec![f64::INFINITY; node_count * node_count];
        let mut via_depot_costs = vec![f64::INFINITY; node_count * node_count];
        let mut return_costs = vec![f64::INFINITY; node_count];
        let mut demands = vec![0.0; customer_count];
        let mut demand_known = vec![false; customer_count];

        direct_costs[depot * node_count + depot] = 0.0;
        return_costs[depot] = 0.0;

        for operator in task.get_operators() {
            let parts: Vec<&str> = operator.name().split_whitespace().collect();
            let metric_cost = usable_cost(
                metric_operator_cost_from_initial_values(task, operator),
            );

            if parts.len() == 3 && parts[0] == "visit" {
                if let (Some(&from), Some(&to)) =
                    (node_by_name.get(parts[1]), node_by_name.get(parts[2]))
                {
                    if let Some(cost) = metric_cost {
                        direct_costs[from * node_count + to] = cost;
                    }

                    if to < customer_count {
                        if let Some(load_id) = load_var_id {
                            for assignment in operator.assignment_effects() {
                                if assignment.affected_var_id() == load_id {
                                    let rhs_id = assignment.var_id();
                                    if let Some(value) = initial_numeric.get(rhs_id).copied() {
                                        if value.is_finite() && value >= 0.0 {
                                            demands[to] = value;
                                            demand_known[to] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else if parts.len() == 2 && parts[0] == "return-to-depot" {
                if let Some(&from) = node_by_name.get(parts[1]) {
                    if let Some(cost) = metric_cost {
                        direct_costs[from * node_count + depot] = cost;
                        return_costs[from] = cost;
                    }
                }
            } else if parts.len() == 3 && parts[0] == "visit-via-depot" {
                if let (Some(&from), Some(&to)) =
                    (node_by_name.get(parts[1]), node_by_name.get(parts[2]))
                {
                    if let Some(cost) = metric_cost {
                        via_depot_costs[from * node_count + to] = cost;
                    }

                    if to < customer_count {
                        if let Some(load_id) = load_var_id {
                            for assignment in operator.assignment_effects() {
                                if assignment.affected_var_id() == load_id {
                                    let rhs_id = assignment.var_id();
                                    if let Some(value) = initial_numeric.get(rhs_id).copied() {
                                        if value.is_finite() && value >= 0.0 {
                                            demands[to] = value;
                                            demand_known[to] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for customer in 0..customer_count {
            let stored = direct_costs[customer * node_count + depot];
            if stored.is_finite() {
                return_costs[customer] = stored;
            }
        }

        let demands_complete = demand_known.iter().all(|known| *known);

        Ok(Self {
            customer_count,
            depot,
            visited_facts,
            location_facts,
            direct_costs,
            via_depot_costs,
            return_costs,
            demands,
            demands_complete,
            capacity,
            load_var_id,
            scratch: RefCell::new(ScratchBuffers {
                propositional: vec![0; task.variables().len()],
                numeric: vec![0.0; task.numeric_variables().len()],
                remaining: Vec::with_capacity(customer_count),
                selected: vec![false; customer_count],
            }),
        })
    }

    fn direct_cost(&self, from: usize, to: usize) -> Option<f64> {
        let node_count = self.customer_count + 1;
        usable_cost(self.direct_costs[from * node_count + to])
    }

    fn via_cost(&self, from: usize, to: usize) -> Option<f64> {
        let node_count = self.customer_count + 1;
        usable_cost(self.via_depot_costs[from * node_count + to])
    }

    fn return_cost(&self, from: usize) -> f64 {
        usable_cost(self.return_costs[from]).unwrap_or(0.0)
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

        state.fill_state(registry, &mut scratch.propositional);

        let resource_enabled = self.capacity.is_some()
            && self.load_var_id.is_some()
            && self.demands_complete;

        if resource_enabled {
            registry
                .view(state)
                .fill_numeric(&mut scratch.numeric)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!("{err:?}"))
                })?;
        }

        let mut current = self.depot;
        for location in &self.location_facts {
            if scratch.propositional[location.fact.var_id] == location.fact.value {
                current = location.node;
                break;
            }
        }

        scratch.remaining.clear();
        for selected in &mut scratch.selected {
            *selected = false;
        }

        for customer in 0..self.customer_count {
            let fact = self.visited_facts[customer];
            if scratch.propositional[fact.var_id] != fact.value {
                scratch.remaining.push(customer);
            }
        }

        if scratch.remaining.is_empty() {
            return Ok(self.return_cost(current));
        }

        let capacity = self.capacity.unwrap_or(f64::MAX);
        let mut simulated_load = if resource_enabled {
            let load_id = self.load_var_id.unwrap_or(0);
            let value = scratch.numeric.get(load_id).copied().unwrap_or(0.0);
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut remaining_count = scratch.remaining.len();
        let mut estimate = 0.0;

        while remaining_count > 0 {
            let mut best_customer: Option<usize> = None;
            let mut best_cost = f64::INFINITY;
            let mut best_uses_depot = false;

            for &customer in &scratch.remaining {
                if scratch.selected[customer] {
                    continue;
                }

                let demand = if resource_enabled {
                    self.demands[customer]
                } else {
                    0.0
                };

                let direct_feasible = !resource_enabled
                    || current == self.depot
                    || simulated_load + demand <= capacity + 1.0e-9;

                if direct_feasible {
                    if let Some(cost) = self.direct_cost(current, customer) {
                        if cost < best_cost {
                            best_cost = cost;
                            best_customer = Some(customer);
                            best_uses_depot = false;
                        }
                    }
                }

                if current != self.depot {
                    if let Some(cost) = self.via_cost(current, customer) {
                        if cost < best_cost {
                            best_cost = cost;
                            best_customer = Some(customer);
                            best_uses_depot = true;
                        }
                    }
                }
            }

            if best_customer.is_none() {
                for &customer in &scratch.remaining {
                    if !scratch.selected[customer] {
                        best_customer = Some(customer);
                        best_cost = 0.0;
                        best_uses_depot = current != self.depot;
                        break;
                    }
                }
            }

            let customer = match best_customer {
                Some(customer) => customer,
                None => break,
            };

            if best_cost.is_finite() && best_cost >= 0.0 {
                estimate += best_cost;
            }

            if resource_enabled {
                if current == self.depot || best_uses_depot {
                    simulated_load = self.demands[customer];
                } else {
                    simulated_load += self.demands[customer];
                }
            }

            scratch.selected[customer] = true;
            remaining_count -= 1;
            current = customer;
        }

        estimate += self.return_cost(current);

        if estimate.is_finite() && estimate >= 0.0 {
            Ok(estimate)
        } else {
            Ok(0.0)
        }
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
