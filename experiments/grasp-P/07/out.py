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

    route_load = 0.0
    radial_sum = 0.0
    radial_count = 0
    for node in route:
        route_load += float(demands[node])
        radial_sum += distances[depot][node]
        radial_count += 1

    candidate_demand = float(demands[customer])
    residual = capacity - route_load - candidate_demand
    if residual < 0.0:
        # Only protects against negligible arithmetic inconsistencies; the
        # contract states that the candidate is feasible.
        residual = 0.0

    total_compatible_demand = 0.0
    largest_compatible_demand = 0.0
    nearest_compatible = math.inf

    for node in unassigned:
        radial_sum += distances[depot][node]
        radial_count += 1

        if node == customer:
            continue

        demand = float(demands[node])
        if 0.0 < demand <= residual:
            total_compatible_demand += demand
            if demand > largest_compatible_demand:
                largest_compatible_demand = demand
            separation = distances[customer][node]
            if separation < nearest_compatible:
                nearest_compatible = separation

    mean_radius = radial_sum / radial_count if radial_count else 0.0
    scale = max(mean_radius, 1.0e-12)

    # A generalized-insertion saving prevents cheap near-depot customers from
    # systematically displacing remote customers that merge well with this
    # route.
    radial_credit = 0.70 * distances[depot][customer]

    if residual > 0.0:
        if largest_compatible_demand > 0.0:
            # Blend a conservative one-customer fill estimate with an
            # optimistic aggregate fill estimate, without solving knapsack.
            one_step_gap = residual - largest_compatible_demand
            aggregate_gap = max(0.0, residual - total_compatible_demand)
            predicted_gap = 0.35 * one_step_gap + 0.65 * aggregate_gap
            continuation_distance = min(nearest_compatible, 2.0 * scale)
        else:
            predicted_gap = residual
            continuation_distance = scale
    else:
        predicted_gap = 0.0
        continuation_distance = 0.0

    utilization = min(1.0, max(0.0, (route_load + candidate_demand) / capacity))
    capacity_penalty = (
        0.90
        * scale
        * utilization
        * utilization
        * (predicted_gap / capacity)
    )

    # Prefer candidates having a nearby customer that can still accompany
    # them, which promotes geographically coherent capacity filling.
    continuation_penalty = 0.14 * continuation_distance

    score = (
        insertion_delta
        - radial_credit
        + capacity_penalty
        + continuation_penalty
    )

    if not math.isfinite(score):
        return 1.0e300
    return float(max(-1.0e300, min(1.0e300, score)))
