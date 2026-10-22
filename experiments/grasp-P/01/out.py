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
    """Score an insertion by radial savings, packing quality, and urgency."""
    route = solution[route_index]
    depot = instance.depot
    distances = instance.distance_matrix

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )
    depot_distance = distances[depot][customer]

    # Gather the current load and a geometry-dependent distance scale without
    # inspecting completed routes.
    route_load = 0.0
    radial_sum = 0.0
    radial_count = 0
    for node in route:
        route_load += instance.demands[node]
        if node != depot:
            radial_sum += distances[depot][node]
            radial_count += 1

    demand = instance.demands[customer]
    remaining = instance.capacity - route_load - demand
    if remaining < 0.0:
        remaining = 0.0

    # best_gap is the residual capacity after the best single feasible
    # follow-up customer. It is only a cheap packing signal, not a search.
    best_gap = remaining
    nearest_other = float("inf")

    for node in unassigned:
        if node == customer or node == depot:
            continue

        radial_sum += distances[depot][node]
        radial_count += 1

        separation = distances[customer][node]
        if separation < nearest_other:
            nearest_other = separation

        other_demand = instance.demands[node]
        if other_demand <= remaining:
            gap = remaining - other_demand
            if gap < best_gap:
                best_gap = gap

    radial_sum += depot_distance
    radial_count += 1
    scale = radial_sum / radial_count if radial_count else 0.0

    capacity = float(instance.capacity)
    fill_after = 1.0 - remaining / capacity
    if fill_after < 0.0:
        fill_after = 0.0
    elif fill_after > 1.0:
        fill_after = 1.0

    packing_weight = 0.12 + 0.30 * fill_after
    packing_penalty = packing_weight * scale * (best_gap / capacity)

    # A mildly isolated customer is more urgent because dense customers have
    # better chances of being grouped together on a later route. Capping the
    # term prevents isolation from overwhelming insertion compatibility.
    isolation_reward = 0.0
    if nearest_other != float("inf") and scale > 0.0:
        isolation_reward = 0.10 * min(nearest_other, scale)

    score = (
        insertion_delta
        - depot_distance
        + packing_penalty
        - isolation_reward
    )

    if isfinite(score):
        return float(score)
    if score != score:
        return 0.0
    return 1.0e300 if score > 0.0 else -1.0e300
