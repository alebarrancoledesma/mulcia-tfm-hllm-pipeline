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
    .map_err(|err| {
        EvaluationError::ComputationFailed(format!(
            "LLM heuristic failed to read numeric state: {err:?}"
        ))
    })?;
```

The returned vector corresponds to PlanForge numeric-variable IDs.

If your heuristic needs numeric variables frequently, discover their IDs ONCE in the constructor and store those IDs in the heuristic struct.

Do NOT search numeric-variable metadata on every heuristic evaluation.

The initial values of all numeric variables are available through:

```rust
task.get_initial_numeric_state_values()
```

The numeric variable collection is available through:

```rust
task.numeric_variables()
```

Numeric variable types can be inspected with:

```rust
numeric_var.get_type()
```

In particular, the inspected CVRP task contained dynamic variables corresponding to:

```text
PNE vehicles()    type=Regular
PNE load()        type=Regular
PNE total-cost()  type=Cost
```

Their numeric IDs are task dependent and MUST NOT be hardcoded.

Do NOT invent undocumented PlanForge methods in order to obtain numeric-variable information.

If some desired CVRP feature cannot be extracted using APIs explicitly provided in this prompt, the template, or the structural operator information described below, prefer a heuristic design that does not require that feature rather than guessing an API.

## Propositional State Access

The complete propositional state can be read efficiently with:

```rust
state.fill_state(registry, &mut propositional_values);
```

The returned vector contains the current value of each propositional variable.

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

Semantic fact names can be obtained using:

```rust
let fact = ExplicitFact::propositional(var_id, value);
let fact_name = task.get_fact_name(&fact);
```

A safe way to inspect all propositional facts during construction is:

```rust
for var_id in 0..task.variables().len() {
    for value in 0..task.variables()[var_id].domain_size() {
        let fact = ExplicitFact::propositional(var_id, value);
        let fact_name = task.get_fact_name(&fact);

        // inspect fact_name
    }
}
```

Perform this kind of semantic discovery only during heuristic construction, not during every state evaluation.

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
task.get_fact_name(&ExplicitFact::propositional(var_id, value))
```

Observed examples:

```text
Atom loc(c10)
NegatedAtom loc(c10)

Atom visited(c7)
NegatedAtom visited(c7)
```

Therefore location and visited mappings can be discovered during heuristic construction by scanning fact names.

When identifying a positive fact such as:

```text
Atom visited(c7)
```

store BOTH:

```text
var_id
positive_value
```

Do NOT assume that propositional value `0` always means true or that value `1` always means false.

For example, a reusable mapping may conceptually contain:

```rust
struct PropFactRef {
    var_id: usize,
    positive_value: usize,
}
```

Then, after using `fill_state()`, a positive fact holds when:

```rust
propositional_values[fact_ref.var_id] == fact_ref.positive_value
```

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

Translator-generated numeric names may contain complex derived expressions.

Do NOT build the heuristic around parsing arbitrary strings such as:

```text
PNE derived!...
```

Prefer structural task/operator information whenever possible.

If demands, capacity or maximum vehicle count cannot be extracted robustly using the documented API, they are OPTIONAL features. Do not implement fragile parsing merely to obtain them.

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

The metric cost of a grounded operator can be obtained with:

```rust
metric_operator_cost_from_initial_values(task, op)
```

This is the preferred way to reconstruct routing costs without parsing translator-generated `travel-cost` variables.

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

If required static information cannot be discovered reliably during `LlmHeuristic::new(task)`, return a descriptive `HeuristicBuildError` rather than panicking.

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
