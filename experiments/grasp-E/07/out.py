from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def _finite(value: float) -> float:
    """Defensively convert an arithmetic result to a finite float."""
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return math.copysign(1.0e300, value)
    return float(value)


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
    radial_sum = distances[depot][customer]
    for node in route:
        route_load += demands[node]
        radial_sum += distances[depot][node]

    customer_demand = float(demands[customer])
    remaining_after = capacity - route_load - customer_demand
    if remaining_after < 0.0:
        remaining_after = 0.0

    depot_distance = distances[depot][customer]

    # Relative to serving every customer alone, this is the negative saving
    # obtained by attaching the candidate to the current route.
    score = insertion_delta - 2.0 * depot_distance

    # Look for one geographically compatible customer that could still follow
    # this insertion. Pair saving measures how naturally the two customers can
    # share a depot trip; the fill factor mildly favors useful demand pairings.
    best_continuation = 0.0
    best_gap = remaining_after
    for other in unassigned:
        if other == customer:
            continue

        other_demand = float(demands[other])
        if other_demand <= remaining_after:
            gap = remaining_after - other_demand
            if gap < best_gap:
                best_gap = gap

            pair_saving = (
                depot_distance
                + distances[depot][other]
                - distances[customer][other]
            )
            if pair_saving > 0.0:
                if remaining_after > 0.0:
                    fill = other_demand / remaining_after
                    if fill < 0.0:
                        fill = 0.0
                    elif fill > 1.0:
                        fill = 1.0
                else:
                    fill = 0.0
                continuation = pair_saving * (0.75 + 0.25 * fill)
                if continuation > best_continuation:
                    best_continuation = continuation

    score -= 0.16 * best_continuation

    # Penalize residual capacity that cannot be closely filled by even one
    # remaining customer. The route's radial scale keeps this term in distance
    # units and makes it vanish in degenerate zero-distance geometry.
    radial_scale = radial_sum / float(len(route) + 1)
    score += 0.08 * radial_scale * (best_gap / capacity)

    return _finite(score)
