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
    """Lower score means a better feasible insertion candidate."""
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
    depot_distance = matrix[depot][customer]

    load = 0
    route_radius = 0.0
    for node in route:
        load += instance.demands[node]
        radial_distance = matrix[depot][node]
        if radial_distance > route_radius:
            route_radius = radial_distance

    remaining_capacity = instance.capacity - load
    demand_fraction = 0.0
    if remaining_capacity > 0 and instance.demands[customer] > 0:
        demand_fraction = instance.demands[customer] / remaining_capacity
        if demand_fraction > 1.0:
            demand_fraction = 1.0

    if route_radius <= 0.0:
        route_radius = depot_distance

    # The first two terms are the negative saving obtained by merging the
    # customer into this route instead of serving it by a singleton route.
    # The bounded final term mildly favors filling the current route well.
    score = (
        insertion_delta
        - 2.0 * depot_distance
        - 0.5 * route_radius * demand_fraction
    )

    if math.isfinite(score):
        return float(score)
    if score < 0.0:
        return -1.0e300
    return 1.0e300
