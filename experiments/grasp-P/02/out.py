from __future__ import annotations

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

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )
    depot_distance = distances[depot][customer]

    load = 0.0
    nearest_route = depot_distance
    for node in route:
        load += demands[node]
        distance = distances[customer][node]
        if distance < nearest_route:
            nearest_route = distance

    remaining = float(instance.capacity) - load
    residual = remaining - demands[customer]
    if residual < 0.0:
        residual = 0.0

    radial_sum = 0.0
    radial_count = 0
    nearest_other = float("inf")
    largest_next_demand = 0.0
    has_next_fit = False

    for other in unassigned:
        radial_sum += distances[depot][other]
        radial_count += 1

        if other == customer:
            continue

        separation = distances[customer][other]
        if separation < nearest_other:
            nearest_other = separation

        demand = demands[other]
        if demand <= residual and demand >= largest_next_demand:
            largest_next_demand = demand
            has_next_fit = True

    scale = radial_sum / radial_count if radial_count else depot_distance
    if scale <= 0.0:
        scale = nearest_route if nearest_route > 0.0 else 1.0

    # Signed attachment regret: negative when this route is a more natural
    # geometric home for the candidate than any still-unassigned customer.
    if nearest_other == float("inf"):
        attachment = 0.0
    else:
        attachment = nearest_route - nearest_other
        if attachment > scale:
            attachment = scale
        elif attachment < -scale:
            attachment = -scale

    # Large fitting demands are perishable: after several small insertions they
    # may cease to fit even though they are geographically appropriate here.
    if remaining > 0.0:
        demand_fraction = float(demands[customer]) / remaining
        if demand_fraction < 0.0:
            demand_fraction = 0.0
        elif demand_fraction > 1.0:
            demand_fraction = 1.0
    else:
        demand_fraction = 0.0

    # A cheap one-step packing signal. It favors choices whose residual can be
    # closely reduced by another customer, or that close the route with little
    # unused capacity. It deliberately avoids subset-sum lookahead.
    if has_next_fit:
        projected_waste = residual - largest_next_demand
    else:
        projected_waste = residual
    if projected_waste < 0.0:
        projected_waste = 0.0
    capacity_fraction = projected_waste / float(instance.capacity)

    score = (
        insertion_delta
        - 0.85 * depot_distance
        + 0.30 * attachment
        - 0.25 * scale * demand_fraction * demand_fraction
        + 0.25 * scale * capacity_fraction
    )
    return float(score)
