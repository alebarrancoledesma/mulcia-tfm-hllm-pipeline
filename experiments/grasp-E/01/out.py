from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def _insertion_two_opt_gain(
    instance: Instance,
    route: list[int],
    customer: int,
    position: int,
) -> float:
    """Best positive 2-opt gain involving an edge created by the insertion."""
    depot = instance.depot
    expanded = route[:position] + [customer] + route[position:]
    cycle = [depot] + expanded + [depot]
    edge_count = len(cycle) - 1

    if edge_count < 4:
        return 0.0

    new_edges = (position, position + 1)
    distances = instance.distance_matrix
    best_gain = 0.0

    for new_edge in new_edges:
        for other_edge in range(edge_count):
            i = min(new_edge, other_edge)
            j = max(new_edge, other_edge)

            # Adjacent edges cannot define a proper 2-opt reversal. The first
            # and last edges are also adjacent in the closed tour.
            if i == j or j == i + 1 or (i == 0 and j == edge_count - 1):
                continue

            a = cycle[i]
            b = cycle[i + 1]
            u = cycle[j]
            v = cycle[j + 1]
            gain = (
                distances[a][b]
                + distances[u][v]
                - distances[a][u]
                - distances[b][v]
            )
            if gain > best_gain:
                best_gain = gain

    return best_gain


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
    distances = instance.distance_matrix
    depot = instance.depot
    capacity = float(instance.capacity)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )
    if insertion_delta < 0.0 and insertion_delta > -1e-12:
        insertion_delta = 0.0

    # Anticipate one of the improvements that the final 2-opt phase can make.
    two_opt_gain = _insertion_two_opt_gain(
        instance, route, customer, position
    )
    repaired_delta = insertion_delta - two_opt_gain

    load = 0.0
    nearest_route = math.inf
    radial_sum = distances[depot][customer]
    radial_count = 1

    for node in route:
        load += float(instance.demands[node])
        distance_to_candidate = distances[customer][node]
        if distance_to_candidate < nearest_route:
            nearest_route = distance_to_candidate
        radial_sum += distances[depot][node]
        radial_count += 1

    if not math.isfinite(nearest_route):
        nearest_route = distances[depot][customer]

    demand = float(instance.demands[customer])
    residual = max(0.0, capacity - load - demand)
    best_one_step_gap = residual
    nearest_other = math.inf
    nearest_feasible_continuation = math.inf

    for node in unassigned:
        if node == customer:
            continue

        radial_sum += distances[depot][node]
        radial_count += 1
        candidate_distance = distances[customer][node]

        if candidate_distance < nearest_other:
            nearest_other = candidate_distance

        other_demand = float(instance.demands[node])
        if other_demand <= residual + 1e-12:
            if candidate_distance < nearest_feasible_continuation:
                nearest_feasible_continuation = candidate_distance
            gap = max(0.0, residual - other_demand)
            if gap < best_one_step_gap:
                best_one_step_gap = gap

    distance_scale = radial_sum / radial_count if radial_count else 0.0
    depot_distance = distances[depot][customer]

    # Prefer candidates that are genuinely attached to the current route over
    # candidates whose natural cluster is still entirely unassigned.
    detachment_penalty = 0.0
    if math.isfinite(nearest_other):
        detachment_penalty = 0.18 * max(
            0.0, nearest_route - nearest_other
        )

    # Favor residual capacities that can be nearly consumed by one remaining
    # customer. Its influence grows as the route becomes loaded.
    load_fraction = min(1.0, max(0.0, load / capacity))
    packing_weight = 0.18 + 0.22 * load_fraction
    gap_fraction = min(1.0, max(0.0, best_one_step_gap / capacity))
    packing_penalty = packing_weight * distance_scale * gap_fraction

    # A nearby customer that still fits after this insertion is a useful next
    # link in a compact geographic chain.
    continuation_reward = 0.0
    if math.isfinite(nearest_feasible_continuation):
        continuation_reward = 0.08 * max(
            0.0, distance_scale - nearest_feasible_continuation
        )

    score = (
        repaired_delta
        - 0.32 * depot_distance
        + detachment_penalty
        + packing_penalty
        - continuation_reward
    )

    if math.isnan(score):
        return 0.0
    if score > 1e300:
        return 1e300
    if score < -1e300:
        return -1e300
    return float(score)
