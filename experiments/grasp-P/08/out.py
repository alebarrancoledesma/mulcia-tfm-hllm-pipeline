from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


_MAX_FLOAT = 1.0e300


def _finite_score(value: float) -> float:
    """Map any accidental numerical overflow to a finite score."""
    if math.isfinite(value):
        return float(value)
    if value == float("-inf"):
        return -_MAX_FLOAT
    if value == float("inf"):
        return _MAX_FLOAT
    return 0.0


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

    # Exact increase in the current closed route length.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    load = 0.0
    radial_scale = 0.0
    route_distance = distances[depot][customer]

    # Nearest-route distance is insensitive to the present ordering and hence
    # remains informative when a later 2-opt pass can substantially reorder it.
    for node in route:
        load += float(demands[node])
        radial = distances[depot][node]
        if math.isfinite(radial) and radial > radial_scale:
            radial_scale = radial
        separation = distances[node][customer]
        if separation < route_distance:
            route_distance = separation

    customer_radial = distances[depot][customer]
    if math.isfinite(customer_radial) and customer_radial > radial_scale:
        radial_scale = customer_radial

    customer_demand = float(demands[customer])
    residual = capacity - load - customer_demand
    if residual < 0.0:
        residual = 0.0

    # The nearest compatible unassigned customer represents the candidate's
    # best geographic alternative if it is not assigned to this route.
    alternative_distance = customer_radial
    packing_gap = residual

    for other in unassigned:
        if other == customer:
            continue

        radial = distances[depot][other]
        if math.isfinite(radial) and radial > radial_scale:
            radial_scale = radial

        other_demand = float(demands[other])
        if customer_demand + other_demand <= capacity:
            separation = distances[customer][other]
            if separation < alternative_distance:
                alternative_distance = separation

        # One-step capacity lookahead: prefer insertions leaving a residual
        # that another still-unassigned demand can closely fill.
        if other_demand <= residual:
            gap = residual - other_demand
            if gap < packing_gap:
                packing_gap = gap

    if not math.isfinite(radial_scale) or radial_scale <= 0.0:
        radial_scale = 1.0

    # Negative values mean the customer is more naturally attached to the
    # current route than to its best compatible alternative cluster.
    assignment_opportunity = route_distance - alternative_distance

    used_after = load + customer_demand
    utilization_pressure = used_after / capacity
    if utilization_pressure < 0.0:
        utilization_pressure = 0.0
    elif utilization_pressure > 1.0:
        utilization_pressure = 1.0

    packing_penalty = (
        0.28
        * radial_scale
        * utilization_pressure
        * (packing_gap / capacity)
    )

    score = (
        insertion_delta
        - 0.32 * customer_radial
        + 0.22 * assignment_opportunity
        + packing_penalty
    )
    return _finite_score(score)
