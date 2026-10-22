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
    """Score a fixed insertion using detour, radial urgency, and capacity fit."""
    route = solution[route_index]
    matrix = instance.distance_matrix
    depot = instance.depot

    previous = depot if position == 0 else route[position - 1]
    following = depot if position == len(route) else route[position]

    insertion_delta = (
        matrix[previous][customer]
        + matrix[customer][following]
        - matrix[previous][following]
    )

    remaining = float(instance.capacity)
    demands = instance.demands
    for node in route:
        remaining -= float(demands[node])

    demand = float(demands[customer])
    if remaining > 0.0:
        fill_fraction = demand / remaining
        if fill_fraction < 0.0:
            fill_fraction = 0.0
        elif fill_fraction > 1.0:
            fill_fraction = 1.0
    else:
        fill_fraction = 1.0

    radial_distance = matrix[depot][customer]
    radial_weight = 0.30 + 0.15 * fill_fraction
    score = float(insertion_delta - radial_weight * radial_distance)

    return score if isfinite(score) else 0.0
