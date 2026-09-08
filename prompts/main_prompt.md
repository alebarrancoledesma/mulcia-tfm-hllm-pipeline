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

Prefer structural task/operator information where possible rather than fragile parsing of translator-generated derived names.

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

This may be useful to reconstruct routing costs without parsing derived `travel-cost` variable names.

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

## Freedom

You MAY:

* add fields to `LlmHeuristic`
* add helper structs and functions
* add scratch buffers
* perform task-dependent precomputation in `new()`
* implement any heuristic algorithm you consider useful for CVRP

You MUST preserve:

* the PlanForge `Heuristic` implementation
* the builder interface
* the external registration mechanism
* generalization across CVRP instances

# PDDL Domain

The following is the PDDL domain for which the heuristic will be generated.

Use it to understand the actions, predicates, numeric fluents, constraints and goal structure of the planning domain.

```pddl
{{ domain_pddl }}
```

# Representative Instances

The following problem instances are representative examples from the target distribution.

Use them to understand structural characteristics of the instances for which the heuristic is being generated.

The final heuristic will be evaluated on unseen instances.

Do NOT memorize particular customers or numerical values from these examples. Instead, infer reusable structural properties that may help guide search on other instances from the same distribution.

{{ representative_instances }}

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

- `code`: the COMPLETE Rust source file implementing the heuristic.
- `what`: a concise technical explanation of WHAT the generated heuristic does, including the information it uses, any task-dependent precomputation, and how the heuristic value is computed from a state.
- `why`: a concise explanation of WHY you selected this heuristic design, including which structural properties of the CVRP and, when applicable, which patterns observed in the representative instances motivated the design.

The `what` and `why` fields are for later interpretability analysis and must accurately describe the generated code.

Return only the JSON object and no additional text.