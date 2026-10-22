from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


_FAR_URGENCY_WEIGHT = 0.15
_FOLLOW_ON_WEIGHT = 0.10
_DEAD_CAPACITY_WEIGHT = 0.35
_FINITE_LIMIT = 1.0e300


def _finite_score(value: float) -> float:
    """Return a finite float even under extreme numeric input."""
    if math.isfinite(value):
        return float(value)
    if math.isnan(value):
        return 0.0
    return _FINITE_LIMIT if value > 0.0 else -_FINITE_LIMIT


def heuristic(
    instance: Instance,
    solution: list[list[int]],
    unassigned: list[int],
    route_index: int,
    customer: int,
    position: int,
) -> float:
    """Score a fixed insertion using geometry, urgency, and capacity cohesion."""
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

    # Mildly prioritize remote customers so that they are less likely to be
    # stranded for a later depot-based route.
    depot_distance = matrix[depot][customer]
    score = insertion_cost - _FAR_URGENCY_WEIGHT * depot_distance

    current_load = 0
    demands = instance.demands
    for node in route:
        current_load += demands[node]

    residual = instance.capacity - current_load - demands[customer]
    if residual < 0:
        residual = 0

    # Prefer candidates having a nearby customer that could still accompany
    # them on this route. This affects route composition rather than position.
    nearest_follow_on = math.inf
    for other in unassigned:
        if other == customer or demands[other] > residual:
            continue
        distance = matrix[customer][other]
        if distance < nearest_follow_on:
            nearest_follow_on = distance

    if math.isfinite(nearest_follow_on):
        score += _FOLLOW_ON_WEIGHT * nearest_follow_on
    elif residual > 0:
        # If this insertion necessarily closes the route, penalize unusable
        # residual capacity on the route's natural geometric distance scale.
        scale = max(
            depot_distance,
            matrix[depot][previous],
            matrix[depot][following],
        )
        score += (
            _DEAD_CAPACITY_WEIGHT
            * scale
            * (residual / instance.capacity)
        )

    return _finite_score(float(score))
