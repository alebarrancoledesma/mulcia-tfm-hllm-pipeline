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

    # Exact increase in the current closed route's length.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    load = 0.0
    nearest_route_distance = distances[depot][customer]
    if route:
        nearest_route_distance = math.inf
        for node in route:
            load += demands[node]
            distance = distances[customer][node]
            if distance < nearest_route_distance:
                nearest_route_distance = distance

    remaining_after = capacity - load - demands[customer]
    if remaining_after < 0.0:
        # The caller promises feasibility; tolerate insignificant roundoff.
        remaining_after = 0.0

    depot_distance_sum = 0.0
    depot_distance_count = 0
    best_pair_saving = 0.0
    best_residual = remaining_after
    has_continuation = False
    customer_depot_distance = distances[depot][customer]

    for other in unassigned:
        depot_distance_sum += distances[depot][other]
        depot_distance_count += 1

        if other == customer:
            continue

        other_demand = demands[other]
        if other_demand <= remaining_after:
            has_continuation = True
            residual = remaining_after - other_demand
            if residual < best_residual:
                best_residual = residual

            # Clarke-Wright saving measures whether the two customers point
            # toward the same spatial region relative to the depot.
            pair_saving = (
                customer_depot_distance
                + distances[depot][other]
                - distances[customer][other]
            )
            if pair_saving > best_pair_saving:
                best_pair_saving = pair_saving

    if not has_continuation:
        best_residual = remaining_after

    distance_scale = (
        depot_distance_sum / depot_distance_count
        if depot_distance_count
        else customer_depot_distance
    )

    used_fraction = load / capacity
    if used_fraction < 0.0:
        used_fraction = 0.0
    elif used_fraction > 1.0:
        used_fraction = 1.0

    # Packing pressure is mild early in a route and stronger near closure.
    packing_weight = 0.06 + 0.18 * used_fraction
    packing_penalty = (
        packing_weight * distance_scale * (best_residual / capacity)
    )

    score = (
        insertion_delta
        + 0.08 * nearest_route_distance
        - 0.12 * best_pair_saving
        + packing_penalty
    )

    if math.isfinite(score):
        return float(score)
    if score < 0.0:
        return -1.0e300
    return 1.0e300
