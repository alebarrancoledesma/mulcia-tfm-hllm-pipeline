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
    """Score a fixed insertion by capacity-weighted depot-trip savings."""
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

    # Cost avoided relative to eventually serving this customer on its own.
    depot_distance = matrix[depot][customer]
    saving = max(0.0, 2.0 * depot_distance - insertion_cost)

    load = sum(instance.demands[node] for node in route)
    remaining = instance.capacity - load
    if remaining > 0:
        capacity_fraction = instance.demands[customer] / remaining
        capacity_fraction = min(1.0, max(0.0, capacity_fraction))
    else:
        capacity_fraction = 0.0

    # Demand is only a modest tie-shaping factor and cannot make a
    # geometrically incompatible, zero-saving insertion attractive.
    score = -saving * (1.0 + 0.25 * capacity_fraction)
    return float(score if isfinite(score) else 0.0)
