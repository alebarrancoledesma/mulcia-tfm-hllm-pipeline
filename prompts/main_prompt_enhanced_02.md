# Role

You are a highly-skilled professor in AI planning and a proficient Rust programmer creating a domain-dependent heuristic function for a PDDL domain.

The heuristic function you create will be used to guide a greedy best-first search to solve instances from this domain. Therefore, the heuristic does not need to be admissible.

For a given state, the heuristic function should estimate the **remaining routing cost** required to serve all unvisited customers and reach a valid goal state as accurately as possible, while remaining efficiently computable.

Your objective is **not** mathematical optimality of the estimate. Your objective is to produce a heuristic that is **likely to reduce the number of expanded nodes** during search on unseen instances from the same client distribution.

Next, you will receive the domain definition and representative client instances. Study them to infer reusable structural properties. Then follow the hard design requirements at the end of this prompt.

# Client Instance Family Profile

The representative instances below come from the **target client distribution**. Use them to specialize your heuristic design to this family, without hardcoding particular object names or numeric values.

Typical structural properties of this distribution:

- **Single depot** constant `d1`; customers named `c2`, `c3`, ...
- **Goal**: vehicle at depot and every customer visited.
- **Metric objective**: minimize `total-cost`.
- **Dynamic state dimensions that must influence the heuristic**:
  - current location (`loc`)
  - visited / unvisited customers (`visited`)
  - current load (`load`)
  - vehicles used (`vehicles`)
- **Resource limits**: `capacity`, `max_vehicles`.
- **Routing actions**:
  - direct visit while load permits
  - visit-via-depot when a new vehicle is needed
  - final return-to-depot
- **Travel costs**: explicit pairwise `travel-cost` values in the instance files; often symmetric in practice, but do not assume symmetry unless verified during construction.
- **Scale**: tens of customers (e.g. 32–37 in the examples), a small fleet (e.g. 5 vehicles), capacity on the order of 100, heterogeneous customer demands.

When writing `why`, explicitly mention **which patterns from the representative instances** motivated your design (for example: fleet-limited capacitated routing, explicit metric travel-cost matrix, need to distinguish direct visits from depot reloads).

Do NOT memorize particular customers, demands, or distances from the examples. Generalize the **structure**, not the literals.

# Client Specialization Requirement

Your heuristic MUST be specialized to the structural family shown by the representative instances, but MUST NOT hardcode object names, numeric IDs, customer counts, demands, distances, capacity, or fleet size from those examples.

Before coding, infer reusable patterns from the representative instances, such as:

- typical scale (number of customers, fleet size),
- demand-to-capacity ratio,
- whether direct visits dominate or depot reloads are frequent,
- whether travel costs are explicit and dense,
- whether customer demands are heterogeneous.

Your design MUST reflect **at least two** of these patterns in the **implementation**, not only in `why`. For example:

- if the fleet is small, include explicit fleet-pressure handling and `visit-via-depot` transitions;
- if demands are heterogeneous, remaining demand and residual load must affect `h(s)`;
- if `n` is tens of customers, prefer `O(n)` or `O(n log n)` evaluation with precomputed structures;
- if travel costs are explicit and dense, reconstruct them from grounded operators rather than from fragile PNE parsing alone.

The routing completion used in `compute_heuristic()` MUST depend on the **current state** (current location, current load, vehicles used, and unvisited customers). Do NOT rely on a single fixed customer visit order computed once in `new()` unless you re-anchor or filter that order from the actual current state on every evaluation.

In `why`, cite which representative-instance patterns motivated each major design choice.

In `instance_patterns`, list the reusable structural patterns you inferred from the representative instances.

# PDDL Domain

The following is the PDDL domain for which the heuristic will be generated.

Use it to understand the actions, predicates, numeric fluents, constraints and goal structure of the planning domain.

```pddl
{{ domain_pddl }}
```

# Representative Instances

The following problem instances are representative examples from the target client distribution.

Use them to understand structural characteristics of the instances for which the heuristic is being generated.

The final heuristic will be evaluated on **unseen instances from the same distribution**.

{{ representative_instances }}

---

# PlanForge Heuristic Environment

You are generating a heuristic for PlanForge, a numeric classical planner.

The heuristic will be used on CVRP instances generated from the same CVRP PDDL domain, but with different numbers of customers, demands, distances, capacities and vehicle limits.

Your generated heuristic must therefore **generalize across instances**.

Do NOT hardcode:

* PlanForge variable IDs
* customer names such as `c2`, `c7`, etc.
* number of customers
* capacity
* number of vehicles
* values from the provided example instances

## PlanForge API

These are the symbols that exist. Use whichever your design needs. Do not invent modules, traits, or methods.

The template already starts with this import; keep it:

```rust
use planforge_sas::numeric_task::{
    AbstractNumericTask,
    ExplicitFact,
    TaskRef,
    metric_operator_cost_from_initial_values,
};
```

`ExplicitFact` and `metric_operator_cost_from_initial_values` live in `planforge_sas::numeric_task`. They are **not** in `planforge_sas::task`, `planforge_sas::state`, or `planforge_search::heuristics`.

### `EvaluationState` (inside `compute_heuristic`)

```rust
eval_state.task()
eval_state.state()
eval_state.state_registry()
eval_state.is_goal()
```

Goal states should return `0.0`.

### Reading a state

```rust
let registry = eval_state.state_registry();
let state = eval_state.state();

registry.view(state).fill_numeric(&mut numeric_values).map_err(|err| {
    EvaluationError::ComputationFailed(format!("{err:?}"))
})?;
state.fill_state(registry, &mut propositional_values);
fact.is_hold(registry.view(state))
```

`fill_numeric` fills a `Vec<f64>` indexed by numeric-variable id and returns `Result`.
`fill_state` fills a `Vec<usize>` indexed by propositional-variable id. A fact `ExplicitFact::propositional(var_id, value)` holds when `propositional_values[var_id] == value`.

If you need the same numeric IDs on every evaluation, resolve them once in `new()` and store them.

### Task (mainly `new()`)

```rust
task.variables()                            // Vec<ExplicitVariable>
task.numeric_variables()                    // Vec<NumericVariable>
task.get_operators()                        // Vec<Operator>
task.get_num_goals()
task.get_goal_fact(i)
task.goals()
task.get_fact_name(&fact)                   // &ExplicitFact
task.get_initial_numeric_state_values()     // &[f64]
task.get_initial_propositional_state_values()
```

`ExplicitVariable`: `name()`, `domain_size()` — no `.values()`.
`NumericVariable`: `name()`, `get_type()` (`Regular` | `Constant` | `Derived` | `Cost`) — no `.id()`. The id is the index:

```rust
for (id, var) in task.numeric_variables().iter().enumerate() {
    let _ = var.name();
}
```

`Operator`: `name()`, `preconditions()`, `effects()`, `assignment_effects()`, `cost()`.
Operator metric cost: `metric_operator_cost_from_initial_values(task, op)`.

Facts:

```rust
let fact = ExplicitFact::propositional(var_id, value);
let fact_name = task.get_fact_name(&fact);
for var_id in 0..task.variables().len() {
    for value in 0..task.variables()[var_id].domain_size() { /* ... */ }
}
```

`HeuristicBuildError` comes from a `String`, not a `&str`. In closures rustc often cannot infer `.to_string().into()` — use an explicit constructor or annotate the return type:

```rust
return Err(HeuristicBuildError::from("missing depot loc fact".to_string()));
// or:
.ok_or_else(|| -> HeuristicBuildError { "missing depot loc fact".to_string().into() })?;
```

Heuristic values are `f64`. Write `let mut x: f64 = 0.0` (or `0.0_f64`) before `.max` / `.min`; a bare `0.0` is an ambiguous `{float}` and does not compile.

### Numeric names: `load`/`vehicles` are safe, `capacity`/`demand`/`max_vehicles` are not

Propositional `variable.name()` is generic (`var0`, …). Semantics are in `get_fact_name`, e.g. `Atom loc(c10)`, `Atom visited(c7)`. Do not assume value `0` means true.

Numeric names look like `PNE load()`, `PNE vehicles()`, `PNE total-cost()`. `load()` and `vehicles()` are safe to find by exact zero-arity name: operators assign/increase them directly, so the translator always keeps them as their own numeric variable.

`capacity()`, `demand(c7)`, and `max_vehicles()` are different: no operator ever assigns them, they are only *read* inside a compound comparison (e.g. `(load) + (demand ?to) <= (capacity)`). The translator is free to fold that whole comparison into one synthesized `PNE derived!...` expression and drop the original `capacity()` / `demand(c7)` variable if nothing else references it by name. Two consequences:

- Exact-name lookup for `capacity()` / `demand(c7)` / `max_vehicles()` can legitimately find **nothing**. This is expected, not a bug, and MUST NOT fail `new()`.
- Substring / `contains` / `ends_with` matching against `derived!...` names is **actively unsafe**, not just fragile: a folded name can contain the text `capacity` or `demand(c7)` as a fragment of a larger expression (e.g. "capacity minus this customer's demand"), so matching on it can silently return a plausible-looking but **wrong** number instead of failing loudly.

Treat `capacity`, `demand`, and `max_vehicles` as optional, best-effort signals, not required inputs:

- look them up only with an exact match of the full trimmed name (e.g. `name.trim() == "capacity()"`), never `contains` / `ends_with` / `starts_with`;
- if the exact name is not found, degrade the resource term of `h` with a permissive default (`f64::MAX` for `capacity`/`max_vehicles`, `0.0` for `demand`) instead of returning `Err` from `new()`;
- never let the absence of `capacity`/`demand`/`max_vehicles` prevent `h_routing` from being computed — `new()` returning `Ok` with a degraded resource term is always correct; returning `Err` because one of these three was absent by exact name is always wrong.

If you look something up by exact name and it is missing: do not invent an API. Either use another documented source (operators, other Regular/Cost PNEs) or drop that ingredient.

Do not hardcode PlanForge IDs, object names, or numeric values from the example instances.

## Grounded Operators

Grounded operator names preserve useful CVRP semantics.

Observed forms include:

```text
visit d1 c7
visit c4 c7
return-to-depot c7
visit-via-depot c4 c7
```

Operators can be inspected using:

```rust
task.get_operators()
op.name()
op.preconditions()
op.effects()
op.assignment_effects()
```

The metric cost of a grounded operator is obtained with (see PlanForge API):

```rust
metric_operator_cost_from_initial_values(task, op)
```

This is the **preferred** way to reconstruct routing costs without relying on fragile parsing of derived `travel-cost` variable names.

## Construction VS Evaluation

Your heuristic object may contain:

* precomputed arrays
* mappings
* helper structs
* cached task information
* reusable scratch buffers

Use `LlmHeuristic::new(task)` for information that depends only on the task.

`compute_heuristic()` may be called hundreds of thousands or millions of times, so it should be relatively cheap.

Do NOT repeatedly:

* scan every task variable by name
* reconstruct static distance matrices
* parse the complete task
* allocate large temporary structures unnecessarily

---

# Hard Design Requirements

These are mandatory. A submission that violates them is invalid even if it compiles.

## MUST

1. Discover `loc` and `visited` facts by scanning semantic fact names via `task.get_fact_name(...)`.
2. Discover numeric IDs for `load()` and `vehicles()` once in `new()`; never search by name in `compute_heuristic()`.
3. Extract travel costs primarily from **grounded operators** using `metric_operator_cost_from_initial_values(task, op)` for operators such as `visit ...` and `visit-via-depot ...`.
4. Implement the heuristic as a **sum of at least two explicit components**, for example:
   - `h(s) = h_routing(s) + h_capacity(s)` or
   - `h(s) = h_routing(s) + h_capacity(s) + h_vehicles(s)`
5. Include a **resource infeasibility check** using best-effort `capacity`/`demand`/`max_vehicles` values (exact-name lookup with a permissive default when absent, per the numeric-names note above): if remaining demand cannot fit in residual vehicle capacity plus remaining fleet capacity, return a large finite penalty.
6. Reflect the domain distinction between:
   - direct `visit` while capacity permits
   - `visit-via-depot` when a new vehicle is needed
7. Add the **final return-to-depot cost** when customers remain or when all are served but the vehicle is not at the depot.
8. Use reusable scratch buffers for repeated evaluation when inspecting many customers or numeric values.
9. Preserve the PlanForge `Heuristic` implementation, builder interface, external registration mechanism, and heuristic name `llm_001`.
10. Reflect at least **two** structural patterns from the representative client instances in the implementation, as required by the Client Specialization Requirement.
11. Make the routing estimate depend on the **current state**, not on a fixed visit order that ignores the current location, load, vehicles used, or unvisited set.

## MUST NOT

1. Use only unvisited-customer count, goal count, or a constant non-zero value as the main heuristic.
2. Use PNE substring parsing as the **only** source of travel costs.
3. Assume distance symmetry unless verified from extracted operator costs.
4. Hardcode PlanForge IDs, object names, customer counts, capacities, or fleet sizes from the examples.
5. Rebuild the full distance matrix or re-scan the entire task on every `compute_heuristic()` call.
6. Sort large node lists from scratch on every evaluation.
7. Produce a generic interchangeable VRP nearest-neighbor heuristic that would look the same for any routing domain.
8. Use one fixed global customer order in `new()` as the entire routing strategy without re-anchoring it from the current state during evaluation.
9. Return `Err` from `new()` because `capacity()`, `demand(...)`, or `max_vehicles()` was not found by exact name, or because a `contains`/`ends_with`/`starts_with` match against those names was used and failed — their absence is expected and MUST degrade the resource term, not abort construction.

---

# Distance Extraction Protocol

Follow this order during `LlmHeuristic::new(task)`:

**Phase A — Primary (required):**

- Scan grounded operators.
- Parse operator names of the forms `visit <from> <to>` and `visit-via-depot <from> <to>`.
- Store arc costs using `metric_operator_cost_from_initial_values(task, op)`.

**Phase B — Fallback (optional):**

- If some arcs remain missing, attempt to recover costs from numeric variable names containing `travel-cost(`.

**Phase C — Last resort:**

- Fill only still-missing arcs conservatively (for example via depot if both legs are known).
- If the resulting distance matrix is unusable, fail construction with a clear `HeuristicBuildError` message instead of silently using arbitrary constants everywhere.

---

# Technique Catalog

Your design must incorporate **at least two** of the following techniques:

1. **Geometric routing estimate** over unvisited customers (for example: minimum spanning tree, sum of minimum depot connections, or a capacitated greedy completion).
2. **Capacity/resource lower bound** using remaining demand, current load, capacity, and remaining vehicles.
3. **Operator-based travel cost reconstruction** from grounded actions.
4. **Dead-end detection** for resource-infeasible states.
5. **Precomputed nearest-neighbor or sorted adjacency lists** for fast evaluation.

Do not rely on a single nearest-neighbor loop as the entire design.

---

# Performance Budget

Target complexity per `compute_heuristic()` call:

- **O(n log n)** or better, where n is the number of customers.
- Acceptable: one pass over customers, lookups in precomputed arrays, small greedy simulation with pre-sorted neighbor lists.
- Unacceptable: full matrix rebuild, full task rescan, or sorting all nodes from scratch on every call.

Prefer:

```rust
// precomputed once in new()
nearest_from: Vec<Vec<usize>>
scratch_numeric: RefCell<Vec<f64>>
scratch_flags: RefCell<Vec<bool>>
```

---

# Edge Cases Your Code Must Handle

1. Goal state → return `0.0`.
2. All customers visited but not at depot → return travel cost to depot only.
3. At depot with zero load and pending customers → start a route by direct visit.
4. At customer, next customer does not fit in current load → via-depot transition if vehicles remain.
5. Remaining total demand exceeds available fleet capacity → large finite penalty.
6. No feasible move remains under resource limits → large finite penalty, not panic and not `0.0`.

---

# Anti-Patterns

```text
BAD — invalid or weak designs:
- h = number of unvisited customers
- h = 0.0 for all non-goal states
- h = task.get_num_goals() or similar goal-count proxy
- travel costs extracted only from PNE substrings, with no operator fallback
- symmetric distance matrix assumed without checking
- no handling of load / vehicles / capacity at all
- a generic VRP nearest-neighbor heuristic that would look the same for any routing domain
- a fixed customer cycle computed once in new() and reused blindly in every state

GOOD — expected designs:
- h combines routing cost estimate + capacity/vehicle penalties
- travel costs extracted from grounded visit / visit-via-depot operators
- current location, load, vehicles, and unvisited set all affect h(s)
- static task information precomputed in new(); cheap state evaluation
- explicit handling of direct visit vs via-depot route change
- a heuristic whose components exist because this client family is capacitated, fleet-limited, metric-cost, and depot-centered
- completion order or start customer chosen from the current state, not from memorized example literals
```

---

# Reference Design Pattern

The following is **not** code to copy verbatim. It shows the expected structure and reasoning style for a strong CVRP heuristic in PlanForge.

```rust
// CONSTRUCTION (once):
// 1. Scan propositional facts -> map customers to visited/loc ExplicitFact
// 2. Identify depot structurally (has loc, not a visited customer)
// 3. Discover load_id and vehicles_id from simple zero-ary PNE names
// 4. Extract arc costs from grounded operators via metric_operator_cost_from_initial_values
// 5. Precompute distance matrix, nearest-neighbor lists, demands, capacity, max_vehicles

// EVALUATION (many times):
// 1. if goal -> 0.0
// 2. read current location, unvisited customers, load, vehicles
// 3. if resource-infeasible -> LARGE_PENALTY
// 4. h_routing = simulate capacitated completion OR geometric lower bound over unvisited
// 5. h_capacity = penalty if remaining demand exceeds available capacity
// 6. return h_routing + h_capacity (+ optional vehicle penalty)
```

Think step-by-step like a domain-dependent planning heuristic designer:

1. What static structure can be extracted once from the task?
2. Which state features distinguish CVRP search states in this domain?
3. What cheap estimate of remaining **metric routing cost** captures those features?
4. How can operator semantics (`visit` vs `visit-via-depot`) be reflected in the estimate?

---

# Evaluation Rubric

Assume your heuristic will be judged on:

1. **Generalization** — no hardcoded IDs, names, or instance-specific constants.
2. **Informativeness** — different states with different location/load/vehicles/unvisited sets receive different values.
3. **Efficiency** — cheap per-state evaluation with precomputation in `new()`.
4. **Robustness** — operator-based cost extraction with fallback strategy.
5. **Domain alignment** — models capacity, fleet limit, direct visits, via-depot transitions, and final return.
6. **Interpretability** — `what`, `why`, and `design` accurately describe the code.

In `why`, explain why your heuristic should be more informative than:

- (a) unvisited-customer count
- (b) sum of minimum depot distances for unvisited customers
- (c) a purely geometric estimate that ignores capacity and vehicle limits

---

# Minimum Ambition Level

A trivial or nearly constant heuristic is unacceptable.

The implementation must contain substantial non-boilerplate logic in `new()` and `compute_heuristic()`.

Returning `h = remaining_customers`, `h = 0.0`, or a single unmodified counter for all non-goal states is invalid.

---

# Template (IMPORTANT)

Based on the instructions and provided domain information, generate valid code starting from this template:

```rust
{{ heuristic_template }}
```

## Heuristic Name

The template contains the heuristic name:

```rust
name: "llm_001"
```

Keep this exact name unchanged in the generated code.

---

# Output Format

Return a JSON object with exactly the following fields:

- `code`: the COMPLETE Rust source file implementing the heuristic.
- `what`: a concise technical explanation of WHAT the generated heuristic does, including the information it uses, any task-dependent precomputation, and how the heuristic value is computed from a state.
- `why`: a concise explanation of WHY you selected this heuristic design, including which structural properties of the CVRP and which patterns observed in the representative client instances motivated each major design choice.
- `instance_patterns`: a short list of reusable structural patterns inferred from the representative client instances (for example: fleet-limited capacitated routing, dense explicit travel-cost matrix, heterogeneous demands).
- `design`: a short list of the techniques used from the Technique Catalog.
- `complexity`: declared asymptotic cost of `compute_heuristic()` per state.
- `fallbacks`: how the code behaves when operator extraction or PNE fallback is incomplete.
- `known_limitations`: 2–3 honest weaknesses of the design.

The `what`, `why`, `instance_patterns`, `design`, and `known_limitations` fields are for later interpretability analysis and must accurately describe the generated code.

Return only the JSON object and no additional text.

---

# Final Checklist

Before returning the JSON, verify:

1. Goal states return `0.0`.
2. Non-goal states return a finite value for solvable states, and a large finite penalty for detected resource dead-ends.
3. All imports are present and the code compiles against the template interface.
4. Static / task-dependent information is extracted in `new()`, not recomputed on every evaluation.
5. Travel costs are extracted primarily from grounded operators.
6. At least two Technique Catalog items are implemented.
7. The heuristic distinguishes location, load, vehicles, and unvisited customers.
8. Direct visit vs via-depot semantics are reflected in the estimate.
9. No customer names, numeric IDs, or instance sizes are hardcoded.
10. At least two representative-instance patterns are reflected in the implementation, not only in prose.
11. Routing completion depends on the current state rather than on a blindly fixed visit order.
12. `what`, `why`, `instance_patterns`, and `design` match the code exactly.
