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
    """Score a feasible insertion; lower values are preferred."""
    route = solution[route_index]
    depot = instance.depot
    distances = instance.distance_matrix
    demands = instance.demands
    capacity = float(instance.capacity)

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    # Exact increase in the current closed-route length.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    depot_distance = distances[depot][customer]
    route_load = 0.0
    best_route_saving = 0.0
    scale_sum = 0.0
    scale_count = 0

    # Order-independent affinity to the route. This remains informative when
    # the subsequent 2-opt pass can substantially change the current order.
    for node in route:
        route_load += float(demands[node])
        node_depot_distance = distances[depot][node]
        saving = depot_distance + node_depot_distance - distances[customer][node]
        if saving > best_route_saving:
            best_route_saving = saving
        scale_sum += node_depot_distance
        scale_count += 1

    demand = float(demands[customer])
    remaining_before = capacity - route_load
    residual = remaining_before - demand
    if residual < 0.0:
        residual = 0.0

    # One-step capacity compatibility and geometric support among customers
    # that could still join this route after the candidate is inserted.
    best_pack_gap = residual
    best_future_saving = 0.0
    for other in unassigned:
        if other == customer:
            continue

        other_depot_distance = distances[depot][other]
        scale_sum += other_depot_distance
        scale_count += 1

        other_demand = float(demands[other])
        if other_demand <= residual:
            gap = residual - other_demand
            if gap < best_pack_gap:
                best_pack_gap = gap

            saving = (
                depot_distance
                + other_depot_distance
                - distances[customer][other]
            )
            if saving > best_future_saving:
                best_future_saving = saving

    if scale_count:
        distance_scale = scale_sum / scale_count
    elif depot_distance > 0.0:
        distance_scale = depot_distance
    else:
        distance_scale = 1.0

    if capacity > 0.0:
        packing_penalty = 0.20 * distance_scale * (best_pack_gap / capacity)
    else:
        packing_penalty = 0.0

    if remaining_before > 0.0:
        filling_bonus = 0.06 * distance_scale * (demand / remaining_before)
    else:
        filling_bonus = 0.0

    score = (
        insertion_delta
        - 0.22 * best_route_saving
        - 0.08 * best_future_saving
        - 0.10 * depot_distance
        + packing_penalty
        - filling_bonus
    )

    # Preserve the required finite-float contract even under anomalous data.
    score = float(score)
    if score != score:
        return 0.0
    if score > 1.0e300:
        return 1.0e300
    if score < -1.0e300:
        return -1.0e300
    return score
