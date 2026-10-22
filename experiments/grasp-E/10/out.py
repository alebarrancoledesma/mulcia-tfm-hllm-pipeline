from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def _clip(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


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

    # Exact increase in the current closed tour. This is the only
    # position-dependent term, so it directly chooses the insertion slot.
    insertion_delta = (
        float(distances[predecessor][customer])
        + float(distances[customer][successor])
        - float(distances[predecessor][successor])
    )
    if insertion_delta < 0.0:
        insertion_delta = 0.0

    load = 0.0
    radial_sum = 0.0
    radial_count = 0

    # Find the two nearest distinct anchors in the current tour, including
    # the depot. Their connection is a relaxed proxy for what 2-opt may make
    # adjacent after construction.
    nearest_distance = float(distances[customer][depot])
    nearest_node = depot
    second_distance = math.inf
    second_node: int | None = None
    nearest_route_customer = math.inf

    radial_sum += float(distances[depot][depot])
    radial_count += 1

    for node in route:
        load += float(demands[node])
        radial_sum += float(distances[depot][node])
        radial_count += 1

        d = float(distances[customer][node])
        if d < nearest_route_customer:
            nearest_route_customer = d

        if d < nearest_distance:
            second_distance = nearest_distance
            second_node = nearest_node
            nearest_distance = d
            nearest_node = node
        elif d < second_distance:
            second_distance = d
            second_node = node

    if not route:
        nearest_route_customer = float(distances[customer][depot])

    candidate_demand = float(demands[customer])
    remaining_after = capacity - load - candidate_demand
    if remaining_after < 0.0:
        # The caller promises feasibility; absorb harmless floating error.
        remaining_after = 0.0

    fresh_remaining_after = capacity - candidate_demand
    if fresh_remaining_after < 0.0:
        fresh_remaining_after = 0.0

    largest_demand_fitting_after = 0.0
    nearest_compatible = math.inf
    nearest_deferred_partner = math.inf

    for node in unassigned:
        radial_sum += float(distances[depot][node])
        radial_count += 1
        if node == customer:
            continue

        demand = float(demands[node])
        d = float(distances[customer][node])

        if demand <= remaining_after:
            if demand > largest_demand_fitting_after:
                largest_demand_fitting_after = demand
            if d < nearest_compatible:
                nearest_compatible = d
        elif demand <= fresh_remaining_after and d < nearest_deferred_partner:
            # This customer could accompany the candidate on a fresh route,
            # but not after inserting it into the current partially full one.
            nearest_deferred_partner = d

    scale = radial_sum / radial_count if radial_count else 1.0
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0

    if second_node is None:
        relaxed_delta = insertion_delta
    else:
        relaxed_delta = (
            nearest_distance
            + second_distance
            - float(distances[nearest_node][second_node])
        )
        if relaxed_delta < 0.0:
            relaxed_delta = 0.0

    # Blend the actual insertion with a non-adjacent two-anchor proxy. The
    # latter makes customer priority less dependent on the current ordering,
    # which will subsequently be changed by 2-opt.
    geometric_cost = (0.72 * insertion_delta + 0.28 * relaxed_delta) / scale

    # One-step capacity granularity: prefer candidates that leave room for a
    # large fitting demand and hence a smaller plausible terminal remainder.
    projected_waste = remaining_after - largest_demand_fitting_after
    if projected_waste < 0.0:
        projected_waste = 0.0
    waste_cost = projected_waste / capacity

    # If a nearby customer can fit next, reward local continuation.
    continuation_bonus = 0.0
    if math.isfinite(nearest_compatible):
        closeness = 1.0 - nearest_compatible / (2.0 * scale)
        continuation_bonus = 0.10 * _clip(closeness, 0.0, 1.0)

    # Avoid taking a customer away from a close partner that no longer fits
    # this route, unless the candidate has even stronger affinity to the
    # current route. Clipping prevents an isolated outlier from dominating.
    deferred_cost = 0.0
    if math.isfinite(nearest_deferred_partner):
        contrast = (nearest_route_customer - nearest_deferred_partner) / scale
        deferred_cost = 0.18 * _clip(contrast, -2.0, 2.0)

    score = geometric_cost + 0.30 * waste_cost + deferred_cost - continuation_bonus
    if not math.isfinite(score):
        return 1.0e100
    return float(score)
