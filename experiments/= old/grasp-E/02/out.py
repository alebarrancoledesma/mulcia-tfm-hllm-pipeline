from __future__ import annotations

import math
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
    """Score an insertion by its cost relative to a singleton route."""
    route = solution[route_index]
    matrix = instance.distance_matrix
    depot = instance.depot

    previous = depot if position == 0 else route[position - 1]
    following = depot if position == len(route) else route[position]

    insertion_cost = (
        matrix[previous][customer]
        + matrix[customer][following]
        - matrix[previous][following]
    )

    # Serving the customer alone would cost two depot legs. Subtracting that
    # baseline makes the score the negative saving obtained by merging the
    # customer into the current route.
    score = insertion_cost - 2.0 * matrix[depot][customer]

    if not math.isfinite(score):
        return 1.0e300
    return float(score)
