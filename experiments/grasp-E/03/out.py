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
    """Score a capacity-aware, cluster-preserving insertion; lower is better."""
    route = solution[route_index]
    depot = instance.depot
    distances = instance.distance_matrix
    demands = instance.demands
    capacity = float(instance.capacity)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    insertion_cost = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )
    depot_distance = distances[depot][customer]

    load = 0.0
    radial_sum = 0.0
    radial_count = 0
    nearest_route = math.inf

    for node in route:
        load += demands[node]
        if node != depot:
            radial_sum += distances[depot][node]
            radial_count += 1
        distance = distances[customer][node]
        if distance < nearest_route:
            nearest_route = distance

    remaining = max(0.0, capacity - load)
    demand = max(0.0, float(demands[customer]))
    residual = max(0.0, remaining - demand)

    nearest_other = math.inf
    largest_residual_fit = 0.0

    for node in unassigned:
        if node == customer or node == depot:
            continue

        radial_sum += distances[depot][node]
        radial_count += 1

        distance = distances[customer][node]
        if distance < nearest_other:
            nearest_other = distance

        other_demand = max(0.0, float(demands[node]))
        if other_demand <= residual + 1.0e-12 and other_demand > largest_residual_fit:
            largest_residual_fit = other_demand

    # Include the candidate in the characteristic geometric scale.
    radial_sum += depot_distance
    radial_count += 1
    scale = radial_sum / radial_count if radial_count else depot_distance
    scale = max(scale, depot_distance, 1.0e-12)

    if not math.isfinite(nearest_route):
        nearest_route = depot_distance
    if not math.isfinite(nearest_other):
        # With no other customer, the depot represents the alternative future route.
        nearest_other = depot_distance

    # Negative values indicate that the customer belongs more naturally to the
    # current route than to the pool left for future routes.
    affinity_gap = nearest_route - nearest_other
    affinity_gap = max(-scale, min(scale, affinity_gap))

    # One-step capacity residue: prefer insertions that leave room for a large
    # remaining demand, or that close the route with little unused capacity.
    packing_waste = (residual - largest_residual_fit) / capacity
    packing_waste = max(0.0, min(1.0, packing_waste))

    # A large fraction of the currently available capacity is perishable: after
    # several small insertions such a customer may no longer fit this route.
    if remaining > 1.0e-12:
        capacity_urgency = max(0.0, min(1.0, demand / remaining))
    else:
        capacity_urgency = 0.0

    score = (
        insertion_cost
        - 0.32 * depot_distance
        + 0.16 * affinity_gap
        + 0.28 * scale * packing_waste
        - 0.14 * scale * capacity_urgency * capacity_urgency
    )

    if math.isnan(score):
        return 0.0
    if score > 1.0e300:
        return 1.0e300
    if score < -1.0e300:
        return -1.0e300
    return float(score)
