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
    route_size = len(route)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == route_size else route[position]

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    # Credit part of the best immediate 2-opt repair involving a newly
    # introduced edge. A partial credit recognizes the final 2-opt phase
    # without deliberately preferring badly crossed partial routes.
    best_repair = 0.0
    new_edges = (
        (predecessor, customer),
        (customer, successor),
    )

    for edge_index in range(route_size + 1):
        if edge_index == position:
            continue  # This old edge is removed by the insertion.

        old_start = depot if edge_index == 0 else route[edge_index - 1]
        old_end = depot if edge_index == route_size else route[edge_index]

        for new_start, new_end in new_edges:
            # A proper 2-opt exchange requires vertex-disjoint edges.
            if (
                old_start == new_start
                or old_start == new_end
                or old_end == new_start
                or old_end == new_end
            ):
                continue

            gain = (
                distances[new_start][new_end]
                + distances[old_start][old_end]
                - distances[new_start][old_start]
                - distances[new_end][old_end]
            )
            if gain > best_repair:
                best_repair = gain

    current_load = 0.0
    radial_sum = 0.0
    radial_count = 0

    for node in route:
        current_load += demands[node]
        radial_sum += distances[depot][node]
        radial_count += 1

    for node in unassigned:
        radial_sum += distances[depot][node]
        radial_count += 1

    distance_scale = radial_sum / radial_count if radial_count else 0.0
    residual = capacity - current_load - demands[customer]
    tolerance = 1.0e-12 * max(1.0, capacity)
    if residual < 0.0 and residual >= -tolerance:
        residual = 0.0

    # Compare the candidate's strongest pair affinity that can still be
    # accommodated on this route with its strongest affinity that would be
    # forced onto another route. Pair affinity is Clarke-Wright-style depot
    # saving and therefore emphasizes close, remote customer pairs.
    available_affinity = 0.0
    blocked_affinity = 0.0
    another_customer_fits = False
    depot_to_customer = distances[depot][customer]

    for other in unassigned:
        if other == customer:
            continue

        affinity = (
            depot_to_customer
            + distances[depot][other]
            - distances[customer][other]
        )
        if affinity < 0.0:
            affinity = 0.0

        if demands[other] <= residual + tolerance:
            another_customer_fits = True
            if affinity > available_affinity:
                available_affinity = affinity
        elif affinity > blocked_affinity:
            blocked_affinity = affinity

    split_penalty = 0.14 * max(0.0, blocked_affinity - available_affinity)

    # If this insertion necessarily closes the route, charge for capacity
    # that will become unusable. The geometric scale keeps this commensurate
    # with distance while an exact fill receives no penalty.
    closure_waste = 0.0
    if not another_customer_fits and capacity > 0.0:
        waste_fraction = max(0.0, residual) / capacity
        closure_waste = 0.30 * distance_scale * waste_fraction

    score = (
        insertion_delta
        - 0.45 * best_repair
        + split_penalty
        + closure_waste
    )

    if math.isfinite(score):
        return float(score)
    if math.isnan(score):
        return 0.0
    return 1.0e300 if score > 0.0 else -1.0e300
