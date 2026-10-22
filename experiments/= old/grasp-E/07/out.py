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
    """Score a candidate by capacity-adjusted depot-trip savings."""
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
    singleton_cost = 2.0 * matrix[depot][customer]

    if not isfinite(insertion_cost) or not isfinite(singleton_cost):
        return 0.0

    saving = singleton_cost - insertion_cost
    if saving <= 0.0:
        return 0.0

    route_load = sum(instance.demands[node] for node in route)
    remaining = instance.capacity - route_load
    demand = instance.demands[customer]

    pressure = 0.0
    if remaining > 0 and demand > 0:
        pressure = float(demand) / float(remaining)
        # The candidate is known to fit; clamping also handles minor bad data.
        if pressure > 1.0:
            pressure = 1.0

    score = -saving / (1.0 + pressure)
    return float(score) if isfinite(score) else 0.0
