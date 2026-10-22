from __future__ import annotations

from math import isfinite
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
    """Score an insertion by its saving over a standalone depot trip."""
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

    if route:
        standalone_cost = (
            matrix[depot][customer] + matrix[customer][depot]
        )
        score = insertion_cost - standalone_cost
    else:
        # No consolidation saving exists for an empty route.
        score = insertion_cost

    result = float(score)
    return result if isfinite(result) else 0.0
