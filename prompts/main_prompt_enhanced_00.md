# Role

You are a highly-skilled professor in AI planning and a proficient Rust programmer creating a domain-dependent heuristic function for a PDDL domain.

The heuristic function you create will be used to guide a search to solve instances from this domain. Therefore, the heuristic does not need to be admissible.

For a given state, the heuristic function should estimate the remaining routing cost required to serve all unvisited customers and reach a valid goal state as accurately as possible, while remaining efficiently computable.

It should minimize the number of expanded nodes during the search.

Next, you will receive a sequence of file contents to help you with your task and to show you the definition of the domain.

# Heuristic Environment

You are generating a heuristic for PlanForge, a numeric classical planner.

The heuristic will be used on CVRP instances generated from the same CVRP PDDL domain, but with different numbers of customers, demands, distances, capacities and vehicle limits.

Your generated heuristic must therefore generalize across instances.

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

For example, under this CVRP domain:

```text
visit A B
```

represents movement from `A` to `B`, so its metric operator cost can be used as the corresponding directed routing cost.

Likewise:

```text
return-to-depot A
```

provides the routing cost from `A` back to the depot.

Do NOT assume that the routing-cost matrix is symmetric.

Operator names may be parsed during heuristic construction to associate grounded operators with CVRP nodes.

Do NOT repeatedly parse operator names during `compute_heuristic()`.

## Numeric Operator Effects

Numeric operator effects can be inspected through:

```rust
op.assignment_effects()
```

Each assignment effect exposes at least:

```rust
assign.affected_var_id()
assign.var_id()
assign.operation()
assign.conditions()
```

The initial value of the numeric variable referenced by the right-hand side is available through:

```rust
task.get_initial_numeric_state_values()[assign.var_id()]
```

This structural information may be used during construction to recover useful task information when the relationship is clear.

For example, a `visit` operator may update a dynamic numeric variable using a constant corresponding to customer demand.

However, do NOT assume a particular assignment-effect pattern unless it can be established from the operators of the current task.

Do NOT rely on fixed numeric IDs or fixed translator ordering.

## Accumulated Cost

`total-cost` represents accumulated path cost.

Do NOT use accumulated `total-cost` as a substitute for the remaining-cost heuristic.

The search algorithm already maintains the cost accumulated so far as `g(s)`.

The generated heuristic should estimate remaining cost:

```text
h(s)
```

rather than duplicate already accumulated routing cost.

## Construction VS Evaluation

Your heuristic object may contain:

* precomputed arrays
* mappings
* helper structs
* cached task information
* reusable scratch buffers

Use `LlmHeuristic::new(task)` for information that depends only on the task.

The constructor is the appropriate place to:

* discover customers and locations
* discover propositional fact mappings
* inspect numeric variables
* inspect grounded operators
* inspect numeric operator effects
* reconstruct static routing costs
* precompute task-dependent structures
* allocate reusable buffers

`compute_heuristic()` may be called hundreds of thousands or millions of times, so it should be relatively cheap.

During `compute_heuristic()`, prefer:

```text
fill_state
fill_numeric
cheap lookup through precomputed mappings
heuristic calculation
```

Do NOT repeatedly:

* scan every task variable by name
* scan every grounded operator
* reconstruct static distance matrices
* parse the complete task
* parse operator names
* allocate large temporary structures unnecessarily

## Mutable Scratch Buffers

The `Heuristic` interface provides `compute_heuristic()` with `&self`.

If reusable mutable scratch memory is required, it may be stored using interior mutability, for example:

```rust
use std::cell::RefCell;

struct ScratchBuffers {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
    // additional reusable buffers...
}

struct LlmHeuristic {
    // precomputed task information...
    scratch: RefCell<ScratchBuffers>,
}
```

Then reusable buffers can be borrowed during evaluation using:

```rust
let mut scratch = self.scratch.borrow_mut();
```

Use this pattern when useful rather than allocating large temporary vectors on every heuristic call.

## Runtime Safety

The generated heuristic must be robust when executed on unseen instances from the same domain.

Return a finite, non-negative `f64`.

Avoid producing:

```text
NaN
+infinity
-infinity
negative heuristic values
```

Guard numerical operations such as division and operations on empty collections.

Do NOT use `unwrap()` or `expect()` on task-dependent or state-dependent data unless the condition has been explicitly guaranteed during construction.

If information that *your* design truly needs cannot be discovered reliably during `LlmHeuristic::new(task)`, return a descriptive `HeuristicBuildError` rather than panicking. A missing exact fluent name (`capacity()`, `demand(...)`, …) is not by itself a reason to fail construction — see PlanForge API.

For this experiment, do NOT declare a state to be a dead end unless impossibility is logically certain.

Prefer a finite heuristic estimate over incorrectly pruning a valid state.

## API Safety

Use only:

* Rust standard-library APIs
* PlanForge APIs already present in the provided template
* PlanForge APIs explicitly documented in this prompt

You MAY add imports from the Rust standard library and from the already available PlanForge crates when required by the implementation.

Do NOT invent PlanForge methods, traits, fields or constructors.

Do NOT assume that an API exists because a similarly named method would be conventional in Rust.

If an optional heuristic idea requires unavailable or uncertain API functionality, choose another implementation that can be expressed using the documented interfaces.

Compilation and correct execution are more important than using every potentially available CVRP feature.

## Freedom

You MAY:

* add fields to `LlmHeuristic`
* add helper structs and functions
* add scratch buffers
* add required imports
* perform task-dependent precomputation in `new()`
* inspect grounded operators and their effects during construction
* implement any heuristic algorithm you consider useful for CVRP

You MUST preserve:

* the PlanForge `Heuristic` implementation
* the builder interface
* the external registration mechanism
* generalization across CVRP instances

Do not modify unrelated planner infrastructure.

Do not implement preferred/helpful operators. This experiment concerns only the scalar heuristic value `h(s)`.

# PDDL Domain

The following is the PDDL domain for which the heuristic will be generated.

Use it to understand the actions, predicates, numeric fluents, constraints and goal structure of the planning domain.

```pddl
{{ domain_pddl }}
```

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

# Output Format

Return a JSON object with exactly the following fields:

* `code`: the COMPLETE Rust source file implementing the heuristic.
* `what`: a concise technical explanation of WHAT the generated heuristic does, including the information it uses, any task-dependent precomputation, and how the heuristic value is computed from a state.
* `why`: a concise explanation of WHY you selected this heuristic design, including which structural properties of the CVRP and, when applicable, which patterns observed in the representative instances motivated the design.

The `what` and `why` fields are for later interpretability analysis and must accurately describe the generated code.

Do not describe techniques or design decisions that are not actually present in the generated code.

Return only the JSON object and no additional text.


# Representative Instances

The following problem instances are representative examples from the target distribution.

Use them to understand structural characteristics of the instances for which the heuristic is being generated.

The final heuristic will be evaluated on unseen instances.

Do NOT memorize particular customers or numerical values from these examples. Instead, infer reusable structural properties that may help guide search on other instances from the same distribution.

{{ representative_instances }}
