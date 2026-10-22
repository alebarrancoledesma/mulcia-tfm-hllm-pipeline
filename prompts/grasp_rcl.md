# Role and objective

You are a highly-skilled professor in combinatorial optimization and a proficient Python programmer designing a CVRP insertion score for GRASP. Produce one useful heuristic to minimize final total routing distance after construction and intra-route 2-opt. Use the representative instances as structural motivation: their layout, demand pattern, and capacity pressure. Be creative with the information the caller gives you. Geometry, demand, remaining capacity, the current route, and the other unassigned customers are all available. This is not a remaining-cost estimate or an admissible search heuristic. Lower score means a better candidate.

# GRASP environment and exact contract

Implement exactly:
`heuristic(instance, solution, unassigned, route_index, customer, position) -> float`.
The caller passes all six arguments by keyword. Return a finite Python float.

Construction opens a new route with the farthest unassigned customer from the depot. For each customer that fits the current route's remaining capacity, the caller evaluates your score at every position in [0, len(route)] and keeps the earliest position whose score is lowest. That customer then enters the RCL with the kept score. The RCL contains entries with score <= `min_score + alpha*(max_score-min_score)`, and the caller randomly chooses one of them. Only the caller uses randomness. Your score selects both the insertion slot and the customer's priority. When no customer fits, the route is closed and a new one is started. Fleet size is unrestricted: the k in instance names is NOT a vehicle constraint.

Runtime API (do not invent attributes or modules):
- `instance.coords`: list of (x, y) coordinates; `instance.demands`: list of demands.
- `instance.capacity`: positive capacity; `instance.depot`: zero-based depot index.
- `instance.n`: number of nodes including depot; `instance.alpha`: RCL parameter.
- `instance.distance_matrix[i][j]`: precomputed real Euclidean distance (math.hypot), NOT rounded integer CVRPLIB distance. Use this matrix; do not rebuild it.
- `solution` is a list of lists of customer indices, NOT a Solution object. Routes exclude depot nodes. `solution[route_index]` is the current partial route; earlier routes are already completed. Do not scan all routes.
- `unassigned` is a list including `customer` itself and possibly other customers that do not fit. Exclude the candidate from any other-customer statistic.
- `customer` is not yet in a route and is already known to fit the current route.
- `position` is one index in [0, len(route)]. The caller invokes you once for every such index of this customer. Its predecessor/successor is the depot at the corresponding boundary. Be robust to empty routes and singleton unassigned.
- Input VRP node labels 1..n are mapped to indices 0..n-1 by the parser.

# Purity and computational limits

No mutations of inputs, global caches, random calls, network, files, logging, environment access, dynamic execution, or extra libraries. Only math and typing from the standard library are permitted if needed, plus the optional exact `from src.cvrp import Instance`. Prefer postponed annotations and TYPE_CHECKING as in the template so importing the module needs no runtime GRASP import. No other GRASP imports. Do not hardcode customer IDs, names, family names, dimensions, capacities, fleets, or numeric values copied from the examples. You may scan the current route and the unassigned customers, including a combined pass over both. Do not rebuild the distance matrix and do not solve an optimization subproblem inside the score. Guard zero denominators and degenerate geometry.

# Representative instances: family {{ family }}

These are exactly five complete real VRP instances, for structural context only. Only source COMMENT fields revealing optimal/best objective values have been omitted; all other source data, including coordinates, demands and capacity, are unchanged. No .sol solution is supplied. Do not compute an integer-distance optimality gap: our objective uses real Euclidean distances and unrestricted fleet size. Do not assume these examples establish generalization. Later evaluation must include held-out instances from this family AND outside this family. No evaluation has occurred yet; do not claim empirical improvements.

{{ representative_instances }}

# Template

Return a complete importable Python module using this interface. Replace the placeholder with your scoring rule and add pure helper functions within the cost constraints.

```python
{{ heuristic_template }}
```

# Output format

Return only one JSON object with exactly three string fields:
- `code`: COMPLETE Python module, without Markdown fences.
- `what`: concise technical explanation accurately describing the score and cost.
- `why`: motivation for this design and its limitations; no fabricated results.
