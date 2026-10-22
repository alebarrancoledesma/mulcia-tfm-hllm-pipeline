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
    matrix = instance.distance_matrix
    depot = instance.depot

    previous = depot if position == 0 else route[position - 1]
    following = depot if position == len(route) else route[position]

    to_customer = matrix[previous][customer]
    from_customer = matrix[customer][following]
    insertion_delta = (
        to_customer + from_customer - matrix[previous][following]
    )

    capacity = float(instance.capacity)
    route_load = sum(instance.demands[node] for node in route)
    remaining = capacity - route_load - instance.demands[customer]
    if remaining < 0.0:
        remaining = 0.0

    # Look for one customer that could still follow this insertion. The
    # depot-based saving measures whether that customer and the candidate
    # are promising members of the same route. Independently, the smallest
    # one-step leftover provides a cheap capacity-packing signal.
    depot_to_customer = matrix[depot][customer]
    best_saving = 0.0
    best_leftover = remaining
    tolerance = 1e-12 * max(1.0, capacity)

    for other in unassigned:
        if other == customer:
            continue
        demand = instance.demands[other]
        if demand <= remaining + tolerance:
            saving = (
                depot_to_customer
                + matrix[depot][other]
                - matrix[customer][other]
            )
            if saving > best_saving:
                best_saving = saving

            leftover = remaining - demand
            if leftover < 0.0:
                leftover = 0.0
            if leftover < best_leftover:
                best_leftover = leftover

    local_scale = max(
        depot_to_customer,
        0.5 * (to_customer + from_customer),
    )
    unused_fraction = min(1.0, max(0.0, best_leftover / capacity))

    score = (
        insertion_delta
        - 0.25 * best_saving
        + 0.10 * local_scale * unused_fraction
    )

    if math.isfinite(score):
        return float(score)
    if math.isfinite(insertion_delta):
        return float(insertion_delta)
    return 0.0
