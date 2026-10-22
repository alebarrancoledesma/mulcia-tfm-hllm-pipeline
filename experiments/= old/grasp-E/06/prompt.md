# Role and objective

You are a highly-skilled professor in combinatorial optimization and a proficient Python programmer designing a CVRP insertion score for GRASP. Produce one useful, inexpensive heuristic, motivated by the examples, to minimize final total routing distance after construction and intra-route 2-opt. This is not a remaining-cost estimate or an admissible search heuristic. Lower score means a better candidate.

# GRASP environment and exact contract

Implement exactly:
`heuristic(instance, solution, unassigned, route_index, customer, position) -> float`.
The caller passes all six arguments by keyword. Return a finite Python float.

Construction opens a new route with the farthest unassigned customer from the depot. For each customer that fits the current route's remaining capacity, the constructor FIRST computes its cheapest geometric insertion position, minimizing `d[prev][customer] + d[customer][next] - d[prev][next]` (first minimum on ties). This position is FIXED before your function runs. Your score cannot change it. The caller forms the RCL with scores <= `min_score + alpha*(max_score-min_score)` and randomly chooses from that RCL. Only the caller uses randomness. When no customer fits, the route is closed and a new one is started. Fleet size is unrestricted: the k in instance names is NOT a vehicle constraint.

Runtime API (do not invent attributes or modules):
- `instance.coords`: list of (x, y) coordinates; `instance.demands`: list of demands.
- `instance.capacity`: positive capacity; `instance.depot`: zero-based depot index.
- `instance.n`: number of nodes including depot; `instance.alpha`: RCL parameter.
- `instance.distance_matrix[i][j]`: precomputed real Euclidean distance (math.hypot), NOT rounded integer CVRPLIB distance. Use this matrix; do not rebuild it.
- `solution` is a list of lists of customer indices, NOT a Solution object. Routes exclude depot nodes. `solution[route_index]` is the current partial route; earlier routes are already completed. Do not scan all routes.
- `unassigned` is a list including `customer` itself and possibly other customers that do not fit. Exclude the candidate from any other-customer statistic.
- `customer` is not yet in a route and is already known to fit the current route.
- `position` is in [0, len(route)]. Its predecessor/successor is the depot at the corresponding boundary. Be robust to empty routes and singleton unassigned.
- Input VRP node labels 1..n are mapped to indices 0..n-1 by the parser.

# Purity and computational limits

No mutations of inputs, global caches, random calls, network, files, logging, environment access, dynamic execution, or extra libraries. Only math and typing from the standard library are permitted if needed, plus the optional exact `from src.cvrp import Instance`. Prefer postponed annotations and TYPE_CHECKING as in the template so importing the module needs no runtime GRASP import. No other GRASP imports. Do not hardcode customer IDs, names, family names, dimensions, capacities, fleets, or example-specific numeric data. Cost per call should be O(1), O(len(route)), or O(len(unassigned)) at most; no nested scans, sorting, full-instance scans, optimization subproblems, or distance-matrix construction. Guard zero denominators and degenerate geometry.

# Representative instances: family E

These are exactly five complete real VRP instances, for structural context only. Only source COMMENT fields revealing optimal/best objective values have been omitted; all other source data, including coordinates, demands and capacity, are unchanged. No .sol solution is supplied. Do not compute an integer-distance optimality gap: our objective uses real Euclidean distances and unrestricted fleet size. Do not assume these examples establish generalization. Later evaluation must include held-out instances from this family AND outside this family. No evaluation has occurred yet; do not claim empirical improvements.

## E-n23-k3
```text
NAME : E-n23-k3
TYPE : CVRP
DIMENSION : 23
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 4500
NODE_COORD_SECTION
1 266 235
2 295 272
3 301 258
4 309 260
5 217 274
6 218 278
7 282 267
8 242 249
9 230 262
10 249 268
11 256 267
12 265 257
13 267 242
14 259 265
15 315 233
16 329 252
17 318 252
18 329 224
19 267 213
20 275 192
21 303 201
22 208 217
23 326 181
DEMAND_SECTION
1 0
2 125
3 84
4 60
5 500
6 300
7 175
8 350
9 150
10 1100
11 4100
12 225
13 300
14 250
15 500
16 150
17 100
18 250
19 120
20 600
21 500
22 175
23 75
DEPOT_SECTION
 1
 -1
EOF

```

## E-n33-k4
```text
NAME : E-n33-k4
TYPE : CVRP
DIMENSION : 33
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 8000
NODE_COORD_SECTION
1 292 495
2 298 427
3 309 445
4 307 464
5 336 475
6 320 439
7 321 437
8 322 437
9 323 433
10 324 433
11 323 429
12 314 435
13 311 442
14 304 427
15 293 421
16 296 418
17 261 384
18 297 410
19 315 407
20 314 406
21 321 391
22 321 398
23 314 394
24 313 378
25 304 382
26 295 402
27 283 406
28 279 399
29 271 401
30 264 414
31 277 439
32 290 434
33 319 433
DEMAND_SECTION
1 0
2 700
3 400
4 400
5 1200
6 40
7 80
8 2000
9 900
10 600
11 750
12 1500
13 150
14 250
15 1600
16 450
17 700
18 550
19 650
20 200
21 400
22 300
23 1300
24 700
25 750
26 1400
27 4000
28 600
29 1000
30 500
31 2500
32 1700
33 1100
DEPOT_SECTION
 1
 -1
EOF

```

## E-n51-k5
```text
NAME : E-n51-k5
TYPE : CVRP
DIMENSION : 51
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 160
NODE_COORD_SECTION
1 30 40
2 37 52
3 49 49
4 52 64
5 20 26
6 40 30
7 21 47
8 17 63
9 31 62
10 52 33
11 51 21
12 42 41
13 31 32
14 5 25
15 12 42
16 36 16
17 52 41
18 27 23
19 17 33
20 13 13
21 57 58
22 62 42
23 42 57
24 16 57
25 8 52
26 7 38
27 27 68
28 30 48
29 43 67
30 58 48
31 58 27
32 37 69
33 38 46
34 46 10
35 61 33
36 62 63
37 63 69
38 32 22
39 45 35
40 59 15
41 5 6
42 10 17
43 21 10
44 5 64
45 30 15
46 39 10
47 32 39
48 25 32
49 25 55
50 48 28
51 56 37
DEMAND_SECTION
1 0
2 7
3 30
4 16
5 9
6 21
7 15
8 19
9 23
10 11
11 5
12 19
13 29
14 23
15 21
16 10
17 15
18 3
19 41
20 9
21 28
22 8
23 8
24 16
25 10
26 28
27 7
28 15
29 14
30 6
31 19
32 11
33 12
34 23
35 26
36 17
37 6
38 9
39 15
40 14
41 7
42 27
43 13
44 11
45 16
46 10
47 5
48 25
49 17
50 18
51 10
DEPOT_SECTION
 1
 -1
EOF

```

## E-n76-k14
```text
NAME : E-n76-k14
TYPE : CVRP
DIMENSION : 76
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 40 40
2 22 22
3 36 26
4 21 45
5 45 35
6 55 20
7 33 34
8 50 50
9 55 45
10 26 59
11 40 66
12 55 65
13 35 51
14 62 35
15 62 57
16 62 24
17 21 36
18 33 44
19 9 56
20 62 48
21 66 14
22 44 13
23 26 13
24 11 28
25 7 43
26 17 64
27 41 46
28 55 34
29 35 16
30 52 26
31 43 26
32 31 76
33 22 53
34 26 29
35 50 40
36 55 50
37 54 10
38 60 15
39 47 66
40 30 60
41 30 50
42 12 17
43 15 14
44 16 19
45 21 48
46 50 30
47 51 42
48 50 15
49 48 21
50 12 38
51 15 56
52 29 39
53 54 38
54 55 57
55 67 41
56 10 70
57 6 25
58 65 27
59 40 60
60 70 64
61 64 4
62 36 6
63 30 20
64 20 30
65 15 5
66 50 70
67 57 72
68 45 42
69 38 33
70 50 4
71 66 8
72 59 5
73 35 60
74 27 24
75 40 20
76 40 37
DEMAND_SECTION
1 0
2 18
3 26
4 11
5 30
6 21
7 19
8 15
9 16
10 29
11 26
12 37
13 16
14 12
15 31
16 8
17 19
18 20
19 13
20 15
21 22
22 28
23 12
24 6
25 27
26 14
27 18
28 17
29 29
30 13
31 22
32 25
33 28
34 27
35 19
36 10
37 12
38 14
39 24
40 16
41 33
42 15
43 11
44 18
45 17
46 21
47 27
48 19
49 20
50 5
51 22
52 12
53 19
54 22
55 16
56 7
57 26
58 14
59 21
60 24
61 13
62 15
63 18
64 11
65 28
66 9
67 37
68 30
69 10
70 8
71 11
72 3
73 1
74 6
75 10
76 20
DEPOT_SECTION
 1
 -1
EOF

```

## E-n101-k8
```text
NAME : E-n101-k8
TYPE : CVRP
DIMENSION : 101
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 200
NODE_COORD_SECTION
1 35 35
2 41 49
3 35 17
4 55 45
5 55 20
6 15 30
7 25 30
8 20 50
9 10 43
10 55 60
11 30 60
12 20 65
13 50 35
14 30 25
15 15 10
16 30 5
17 10 20
18 5 30
19 20 40
20 15 60
21 45 65
22 45 20
23 45 10
24 55 5
25 65 35
26 65 20
27 45 30
28 35 40
29 41 37
30 64 42
31 40 60
32 31 52
33 35 69
34 53 52
35 65 55
36 63 65
37 2 60
38 20 20
39 5 5
40 60 12
41 40 25
42 42 7
43 24 12
44 23 3
45 11 14
46 6 38
47 2 48
48 8 56
49 13 52
50 6 68
51 47 47
52 49 58
53 27 43
54 37 31
55 57 29
56 63 23
57 53 12
58 32 12
59 36 26
60 21 24
61 17 34
62 12 24
63 24 58
64 27 69
65 15 77
66 62 77
67 49 73
68 67 5
69 56 39
70 37 47
71 37 56
72 57 68
73 47 16
74 44 17
75 46 13
76 49 11
77 49 42
78 53 43
79 61 52
80 57 48
81 56 37
82 55 54
83 15 47
84 14 37
85 11 31
86 16 22
87 4 18
88 28 18
89 26 52
90 26 35
91 31 67
92 15 19
93 22 22
94 18 24
95 26 27
96 25 24
97 22 27
98 25 21
99 19 21
100 20 26
101 18 18
DEMAND_SECTION
1 0
2 10
3 7
4 13
5 19
6 26
7 3
8 5
9 9
10 16
11 16
12 12
13 19
14 23
15 20
16 8
17 19
18 2
19 12
20 17
21 9
22 11
23 18
24 29
25 3
26 6
27 17
28 16
29 16
30 9
31 21
32 27
33 23
34 11
35 14
36 8
37 5
38 8
39 16
40 31
41 9
42 5
43 5
44 7
45 18
46 16
47 1
48 27
49 36
50 30
51 13
52 10
53 9
54 14
55 18
56 2
57 6
58 7
59 18
60 28
61 3
62 13
63 19
64 10
65 9
66 20
67 25
68 25
69 36
70 6
71 5
72 15
73 25
74 9
75 8
76 18
77 13
78 14
79 3
80 23
81 6
82 26
83 16
84 11
85 7
86 41
87 35
88 26
89 9
90 15
91 3
92 1
93 2
94 22
95 27
96 20
97 11
98 12
99 10
100 9
101 17
DEPOT_SECTION
 1
 -1
EOF

```

# Template

Return a complete importable Python module using this interface. You may change the scoring formula and add pure helper functions within the cost constraints.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def heuristic(
    instance: Instance,
    solution: list[list[int]],
    unassigned: list[int],
    route_index: int,
    customer: int,
    position: int,
) -> float:
    """Lower score means a better feasible insertion candidate."""
    # Replace this baseline with your proposed scoring rule.
    route = solution[route_index]
    matrix = instance.distance_matrix
    previous = instance.depot if position == 0 else route[position - 1]
    following = instance.depot if position == len(route) else route[position]
    return float(matrix[previous][customer] + matrix[customer][following]
                 - matrix[previous][following])

```

# Output format

Return only one JSON object with exactly three string fields:
- `code`: COMPLETE Python module, without Markdown fences.
- `what`: concise technical explanation accurately describing the score and cost.
- `why`: motivation for this design and its limitations; no fabricated results.
