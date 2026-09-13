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

## Available State API

Inside `compute_heuristic()`, `EvaluationState` provides:

```rust
eval_state.task()
eval_state.state()
eval_state.state_registry()
eval_state.is_goal()
```

Goal states should return:

```rust
0.0
```

## Numeric State Access

The full numeric state can be read efficiently with:

```rust
let registry = eval_state.state_registry();
let state = eval_state.state();

registry
    .view(state)
    .fill_numeric(&mut numeric_values)
```

The returned vector corresponds to PlanForge numeric-variable IDs.

If your heuristic needs numeric variables frequently, discover their IDs ONCE in the constructor and store those IDs in the heuristic struct.

Do NOT search numeric-variable names on every heuristic evaluation.

## Propositional State Access

The complete propositional state can be read efficiently with:

```rust
state.fill_state(registry, &mut propositional_values);
```

Alternatively, an individual fact can be tested with:

```rust
fact.is_hold(registry.view(state))
```

For heuristics that inspect many customers, prefer reading the complete propositional state once.

## Task Introspection

The task exposes information including:

```rust
task.variables()
task.numeric_variables()
task.get_operators()
task.get_num_goals()
task.get_goal_fact(i)
task.get_fact_name(...)
task.get_initial_numeric_state_values()
```

A propositional fact can be constructed using:

```rust
ExplicitFact::propositional(var_id, value)
```

## IMPORTANT Observations from the CVRP Domain

PlanForge propositional variable names themselves are generic:

```text
var0
var1
var18
...
```

Do NOT use `variable.name()` to identify CVRP propositions.

Semantic fact names ARE preserved through:

```rust
task.get_fact_name(...)
```

Observed examples:

```text
Atom loc(c10)
NegatedAtom loc(c10)

Atom visited(c7)
NegatedAtom visited(c7)
```

Therefore location and visited mappings can be discovered during heuristic construction by scanning fact names.

## Numeric Variables

In the inspected CVRP instance, the dynamic variables were exposed cleanly as:

```text
PNE vehicles()    type=Regular
PNE load()        type=Regular
PNE total-cost()  type=Cost
```

Their numeric IDs are instance/task dependent and MUST NOT be hardcoded.

Many other numeric expressions, especially those involving:

```text
demand
capacity
travel-cost
max-vehicles
```

may be compiled by PlanForge into complex derived PNE expressions.

Do NOT assume they will appear as simple strings such as:

```text
demand(c7)
capacity()
travel-cost(c4,c7)
```

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

PlanForge's FF implementation also uses:

```rust
metric_operator_cost_from_initial_values(task, op)
```

to obtain the metric cost of grounded operators.

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
5. Include a **resource infeasibility check**: if remaining demand cannot fit in residual vehicle capacity plus remaining fleet capacity, return a large finite penalty.
6. Reflect the domain distinction between:
   - direct `visit` while capacity permits
   - `visit-via-depot` when a new vehicle is needed
7. Add the **final return-to-depot cost** when customers remain or when all are served but the vehicle is not at the depot.
8. Use reusable scratch buffers for repeated evaluation when inspecting many customers or numeric values.
9. Preserve the PlanForge `Heuristic` implementation, builder interface, external registration mechanism, and heuristic name `llm_001`.

## MUST NOT

1. Use only unvisited-customer count, goal count, or a constant non-zero value as the main heuristic.
2. Use PNE substring parsing as the **only** source of travel costs.
3. Assume distance symmetry unless verified from extracted operator costs.
4. Hardcode PlanForge IDs, object names, customer counts, capacities, or fleet sizes from the examples.
5. Rebuild the full distance matrix or re-scan the entire task on every `compute_heuristic()` call.
6. Sort large node lists from scratch on every evaluation.

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

GOOD — expected designs:
- h combines routing cost estimate + capacity/vehicle penalties
- travel costs extracted from grounded visit / visit-via-depot operators
- current location, load, vehicles, and unvisited set all affect h(s)
- static task information precomputed in new(); cheap state evaluation
- explicit handling of direct visit vs via-depot route change
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
- `why`: a concise explanation of WHY you selected this heuristic design, including which structural properties of the CVRP and which patterns observed in the representative client instances motivated the design.
- `design`: a short list of the techniques used from the Technique Catalog.
- `complexity`: declared asymptotic cost of `compute_heuristic()` per state.
- `fallbacks`: how the code behaves when operator extraction or PNE fallback is incomplete.
- `known_limitations`: 2–3 honest weaknesses of the design.

The `what`, `why`, `design`, and `known_limitations` fields are for later interpretability analysis and must accurately describe the generated code.

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
10. `what` and `why` match the code exactly.
