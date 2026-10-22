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
    radius_sum = 0.0
    radius_count = 0

    for node in route:
        load += float(demands[node])
        if node != depot:
            radius_sum += distances[depot][node]
            radius_count += 1

    # Obtain a scale that adapts the packing terms to the instance geometry.
    # The candidate is excluded so this statistic does not favor itself.
    for node in unassigned:
        if node != customer and node != depot:
            radius_sum += distances[depot][node]
            radius_count += 1

    if radius_count:
        distance_scale = radius_sum / radius_count
    else:
        distance_scale = distances[depot][customer]
    if not math.isfinite(distance_scale) or distance_scale <= 1e-12:
        distance_scale = 1.0

    residual = capacity - load - float(demands[customer])
    tolerance = 1e-12 * max(1.0, capacity)
    if residual < 0.0 and residual >= -tolerance:
        residual = 0.0

    residual_ratio = residual / capacity
    residual_ratio = min(1.0, max(0.0, residual_ratio))

    # One-step continuation statistic: prefer a customer that has a nearby
    # successor whose demand also packs the resulting residual well.
    best_continuation = math.inf
    for other in unassigned:
        if other == customer or other == depot:
            continue
        other_demand = float(demands[other])
        if other_demand <= residual + tolerance:
            leftover_ratio = max(0.0, residual - other_demand) / capacity
            continuation = (
                distances[customer][other]
                + 0.35 * distance_scale * leftover_ratio * leftover_ratio
            )
            if continuation < best_continuation:
                best_continuation = continuation

    if math.isfinite(best_continuation):
        # Mild pressure toward fuller routes, while preserving a spatially
        # coherent feasible continuation.
        packing_term = 0.10 * distance_scale * residual_ratio * residual_ratio
        continuation_term = 0.10 * best_continuation
    else:
        # This insertion closes the route under the caller's policy. Penalize
        # unused capacity more strongly when no further customer can fit.
        packing_term = 0.35 * distance_scale * residual_ratio * residual_ratio
        continuation_term = 0.0

    score = insertion_delta + packing_term + continuation_term
    if not math.isfinite(score):
        return 1e300
    return float(score)
