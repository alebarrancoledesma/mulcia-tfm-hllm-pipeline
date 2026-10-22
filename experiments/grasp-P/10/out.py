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

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    route_load = 0.0
    for node in route:
        route_load += float(demands[node])

    residual = capacity - route_load - float(demands[customer])
    if residual < 0.0:
        residual = 0.0

    depot_distance = distances[depot][customer]
    radius_sum = depot_distance
    radius_count = 1

    nearest_other = math.inf
    nearest_feasible = math.inf
    best_continuation = math.inf
    has_other = False
    has_feasible_continuation = False

    for other in unassigned:
        if other == customer:
            continue

        has_other = True
        radius_sum += distances[depot][other]
        radius_count += 1

        customer_distance = distances[customer][other]
        if customer_distance < nearest_other:
            nearest_other = customer_distance

        if float(demands[other]) <= residual:
            has_feasible_continuation = True
            if customer_distance < nearest_feasible:
                nearest_feasible = customer_distance

            extension = (
                customer_distance
                + distances[other][depot]
                - depot_distance
            )
            if extension < best_continuation:
                best_continuation = extension

    scale = radius_sum / radius_count
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0

    residual_fraction = residual / capacity if capacity > 0.0 else 0.0
    if residual_fraction < 0.0:
        residual_fraction = 0.0
    elif residual_fraction > 1.0:
        residual_fraction = 1.0

    continuation_term = 0.0
    stranding_term = 0.0
    packing_term = 0.0

    if has_feasible_continuation:
        # Cheap depot-relative continuation favors coherent radial chains.
        continuation_term = max(0.0, best_continuation)

        # Penalize leaving a close customer behind solely because its demand
        # no longer fits, when the nearest fitting continuation is farther away.
        if has_other and math.isfinite(nearest_other):
            stranding_term = max(0.0, nearest_feasible - nearest_other)
            stranding_term *= residual_fraction
    else:
        # If this insertion closes the route, charge for unused capacity and
        # for abandoning a nearby neighbor, but make both vanish when full.
        packing_term = scale * residual_fraction * residual_fraction
        if has_other and math.isfinite(nearest_other):
            stranding_term = max(0.0, scale - nearest_other)
            stranding_term *= residual_fraction

    score = (
        insertion_delta
        - 0.22 * depot_distance
        + 0.14 * continuation_term
        + 0.20 * stranding_term
        + 0.30 * packing_term
    )

    if not math.isfinite(score):
        return 1.0e300
    if score > 1.0e300:
        return 1.0e300
    if score < -1.0e300:
        return -1.0e300
    return float(score)
