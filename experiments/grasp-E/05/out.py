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
    depot = instance.depot
    distances = instance.distance_matrix
    demands = instance.demands
    capacity = float(instance.capacity)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    # Exact increase in the current closed-route length.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    load = 0.0
    radius_sum = distances[depot][customer]
    for node in route:
        load += demands[node]
        radius_sum += distances[depot][node]

    customer_radius = distances[depot][customer]
    residual = capacity - load - demands[customer]
    tolerance = 1.0e-12 * max(1.0, capacity)
    if residual < tolerance:
        residual = 0.0

    # Inspect feasible continuations after this insertion. The largest fitting
    # demand estimates one-step packability, while the closest fitting node
    # estimates whether the route can continue without a geographical jump.
    largest_fitting_demand = 0.0
    nearest_fitting_distance = 0.0
    has_fitting_continuation = False

    for other in unassigned:
        if other == customer:
            continue
        other_demand = float(demands[other])
        if other_demand <= residual + tolerance:
            if not has_fitting_continuation:
                nearest_fitting_distance = distances[customer][other]
                has_fitting_continuation = True
            else:
                nearest_fitting_distance = min(
                    nearest_fitting_distance, distances[customer][other]
                )
            if other_demand > largest_fitting_demand:
                largest_fitting_demand = other_demand

    if has_fitting_continuation:
        projected_waste = max(0.0, residual - largest_fitting_demand) / capacity
    else:
        projected_waste = residual / capacity
        nearest_fitting_distance = 0.0

    # A route-local distance scale makes the capacity term meaningful across
    # coordinate systems without rebuilding geometric distances.
    geometric_scale = max(
        customer_radius,
        radius_sum / (len(route) + 1),
        1.0e-9,
    )

    score = (
        insertion_delta
        - 0.80 * customer_radius
        + 0.12 * nearest_fitting_distance
        + 0.55 * geometric_scale * projected_waste
    )

    if math.isfinite(score):
        return float(score)
    if score > 0.0:
        return 1.0e300
    if score < 0.0:
        return -1.0e300
    return 0.0
