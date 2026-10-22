from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


_DISTANCE_LIMIT = 1.0e100
_SCORE_LIMIT = 1.0e300


def _safe_distance(instance: Instance, i: int, j: int) -> float:
    """Return a bounded, nonnegative distance to keep the score finite."""
    value = float(instance.distance_matrix[i][j])
    if not math.isfinite(value):
        return _DISTANCE_LIMIT
    if value <= 0.0:
        return 0.0
    return min(value, _DISTANCE_LIMIT)


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
    capacity = float(instance.capacity)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    d_pc = _safe_distance(instance, predecessor, customer)
    d_cs = _safe_distance(instance, customer, successor)
    d_ps = _safe_distance(instance, predecessor, successor)
    insertion_delta = d_pc + d_cs - d_ps

    d0c = _safe_distance(instance, depot, customer)
    customer_demand = float(instance.demands[customer])

    route_load = 0.0
    route_bond = 0.0
    radial_sum = d0c

    # Clarke-Wright savings measure whether the candidate belongs to the
    # current route's radial/geometric sector. It is independent of the
    # current ordering, which is useful when 2-opt will subsequently reorder.
    for node in route:
        route_load += float(instance.demands[node])
        d0n = _safe_distance(instance, depot, node)
        radial_sum += d0n
        bond = d0c + d0n - _safe_distance(instance, customer, node)
        if bond > route_bond:
            route_bond = bond

    remaining_after = capacity - route_load - customer_demand
    if not math.isfinite(remaining_after):
        remaining_after = 0.0
    if remaining_after < 0.0:
        remaining_after = 0.0

    joinable_bond = 0.0
    blocked_bond = 0.0
    best_leftover = remaining_after
    has_joinable = False

    # A candidate is attractive when it leaves room for a geographically
    # compatible follow-up and permits tight demand packing. Conversely, a
    # strong partner that would no longer fit is evidence for preserving that
    # pair for a later route.
    for other in unassigned:
        if other == customer:
            continue

        other_demand = float(instance.demands[other])
        d0o = _safe_distance(instance, depot, other)
        bond = d0c + d0o - _safe_distance(instance, customer, other)
        if bond < 0.0:
            bond = 0.0

        if other_demand <= remaining_after:
            has_joinable = True
            if bond > joinable_bond:
                joinable_bond = bond
            leftover = remaining_after - other_demand
            if leftover < best_leftover:
                best_leftover = leftover
        elif bond > blocked_bond:
            blocked_bond = bond

    if not has_joinable:
        best_leftover = remaining_after

    if capacity > 0.0 and math.isfinite(capacity):
        packing_fraction = best_leftover / capacity
        if not math.isfinite(packing_fraction):
            packing_fraction = 0.0
        packing_fraction = min(1.0, max(0.0, packing_fraction))
    else:
        packing_fraction = 0.0

    radial_scale = radial_sum / float(len(route) + 1)
    blocked_excess = max(0.0, blocked_bond - route_bond)

    score = (
        insertion_delta
        - 0.30 * d0c
        - 0.12 * route_bond
        - 0.06 * joinable_bond
        + 0.18 * blocked_excess
        + 0.20 * radial_scale * packing_fraction
    )

    if math.isnan(score):
        return 0.0
    if score >= _SCORE_LIMIT:
        return _SCORE_LIMIT
    if score <= -_SCORE_LIMIT:
        return -_SCORE_LIMIT
    return float(score)
