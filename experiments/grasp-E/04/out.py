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

    # Exact increase in the current closed tour. This determines the slot.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    load = 0.0
    nearest_route = math.inf
    radial_sum = 0.0
    radial_count = 0

    for node in route:
        load += demands[node]
        distance = distances[customer][node]
        if distance < nearest_route:
            nearest_route = distance
        radial_sum += distances[depot][node]
        radial_count += 1

    candidate_radial = distances[depot][customer]
    nearest_other = math.inf
    nearest_feasible_followup = math.inf

    remaining_before = max(0.0, capacity - load)
    remaining_after = max(0.0, remaining_before - demands[customer])
    tolerance = 1.0e-12 * max(1.0, capacity)

    for node in unassigned:
        radial_sum += distances[depot][node]
        radial_count += 1

        if node == customer:
            continue

        distance = distances[customer][node]
        if distance < nearest_other:
            nearest_other = distance
        if demands[node] <= remaining_after + tolerance:
            if distance < nearest_feasible_followup:
                nearest_feasible_followup = distance

    scale = radial_sum / radial_count if radial_count else candidate_radial
    scale = max(scale, candidate_radial, 1.0e-12)

    if not math.isfinite(nearest_route):
        nearest_route = candidate_radial
    if not math.isfinite(nearest_other):
        nearest_other = scale

    # A customer is attractive when the current route is a more natural
    # geometric neighbor than the still-unassigned population.
    affinity_contrast = nearest_route - nearest_other
    affinity_contrast = max(-scale, min(scale, affinity_contrast))

    # Reward sharing a depot excursion with a distant route rather than leaving
    # distant customers for additional routes.
    radial_reward = -0.18 * candidate_radial
    affinity_term = 0.28 * affinity_contrast

    demand = max(0.0, float(demands[customer]))
    if remaining_before > tolerance:
        fill_fraction = min(1.0, demand / remaining_before)
    else:
        fill_fraction = 0.0

    # Mildly prefer insertions that make decisive use of the current residual
    # capacity, especially useful under heterogeneous demand.
    packing_reward = -0.12 * scale * fill_fraction * fill_fraction

    pressure = min(1.0, max(0.0, (load + demand) / capacity))
    continuation_term = 0.0
    closure_penalty = 0.0

    if math.isfinite(nearest_feasible_followup):
        # Under high capacity pressure, preserve a nearby customer that can
        # actually follow the candidate, not merely a nearby oversized one.
        continuation_term = 0.10 * pressure * nearest_feasible_followup
    elif remaining_after > tolerance:
        # The insertion would immediately close the route with unused capacity.
        closure_penalty = (
            0.50
            * scale
            * (remaining_after / capacity)
            * (0.5 + 0.5 * pressure)
        )

    score = (
        insertion_delta
        + radial_reward
        + affinity_term
        + packing_reward
        + continuation_term
        + closure_penalty
    )

    if math.isnan(score):
        return 0.0
    if score > 1.0e300:
        return 1.0e300
    if score < -1.0e300:
        return -1.0e300
    return float(score)
