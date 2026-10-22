from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


_MAX_FLOAT = 1.7976931348623157e308


def heuristic(
    instance: Instance,
    solution: list[list[int]],
    unassigned: list[int],
    route_index: int,
    customer: int,
    position: int,
) -> float:
    """Score a candidate by generalized insertion cost; lower is better."""
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

    # A remote customer is more expensive to leave for a later depot-based
    # route. Subtracting its depot distance adds this radial urgency while the
    # insertion cost still enforces geometric compatibility with this route.
    score = insertion_cost - matrix[depot][customer]

    if isfinite(score):
        return float(score)
    if score < 0.0:
        return -_MAX_FLOAT
    return _MAX_FLOAT
