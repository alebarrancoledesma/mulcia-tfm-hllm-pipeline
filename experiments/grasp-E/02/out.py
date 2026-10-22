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
    """Lower score means a better feasible insertion candidate."""
    route = solution[route_index]
    depot = instance.depot
    distances = instance.distance_matrix
    demands = instance.demands
    capacity = float(instance.capacity)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    # Exact change in the current closed-tour length at this insertion slot.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    load = 0.0
    distance_sum = 0.0
    distance_count = 0
    for node in route:
        load += float(demands[node])
        depot_distance = distances[depot][node]
        if depot_distance > 0.0 and isfinite(depot_distance):
            distance_sum += depot_distance
            distance_count += 1

    for node in unassigned:
        if node == depot:
            continue
        depot_distance = distances[depot][node]
        if depot_distance > 0.0 and isfinite(depot_distance):
            distance_sum += depot_distance
            distance_count += 1

    # A dimensionally consistent scale for capacity-related terms.
    distance_scale = distance_sum / distance_count if distance_count else 1.0

    candidate_demand = max(0.0, float(demands[customer]))
    remaining = max(0.0, capacity - load)
    residual = max(0.0, remaining - candidate_demand)
    load_fraction = min(1.0, max(0.0, load / capacity))
    residual_fraction = min(1.0, residual / capacity)

    depot_distance = distances[depot][customer]

    # Examine whether choosing this customer preserves a geographically useful
    # continuation, or narrowly prevents a close customer from joining.
    best_continuation = 0.0
    best_blocked_affinity = 0.0
    can_continue = False
    tolerance = capacity * 1.0e-12

    for other in unassigned:
        if other == customer or other == depot:
            continue

        other_demand = max(0.0, float(demands[other]))
        if other_demand > remaining + tolerance:
            continue

        affinity = (
            depot_distance
            + distances[depot][other]
            - distances[customer][other]
        )
        if affinity < 0.0 or not isfinite(affinity):
            affinity = 0.0

        if other_demand <= residual + tolerance:
            can_continue = True
            if affinity > best_continuation:
                best_continuation = affinity
        elif affinity > best_blocked_affinity:
            best_blocked_affinity = affinity

    # Filling capacity matters increasingly late in a route. If no customer can
    # use the residual capacity, charge an additional dead-space penalty.
    packing_weight = 0.10 + 0.35 * load_fraction
    packing_penalty = distance_scale * packing_weight * residual_fraction
    if residual > tolerance and not can_continue:
        packing_penalty += 0.22 * distance_scale * residual_fraction

    # Blocking is less objectionable when the insertion nearly fills the route.
    blocking_ratio = residual / remaining if remaining > tolerance else 0.0

    score = (
        insertion_delta
        - 0.90 * depot_distance
        + packing_penalty
        - 0.10 * best_continuation
        + 0.18 * blocking_ratio * best_blocked_affinity
    )

    if not isfinite(score):
        return 0.0
    if score > 1.0e300:
        return 1.0e300
    if score < -1.0e300:
        return -1.0e300
    return float(score)
