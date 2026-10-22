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
    """Score an insertion using detour, radial urgency, and feasible affinity."""
    route = solution[route_index]
    matrix = instance.distance_matrix
    depot = instance.depot

    previous = depot if position == 0 else route[position - 1]
    following = depot if position == len(route) else route[position]

    insertion_cost = (
        matrix[previous][customer]
        + matrix[customer][following]
        - matrix[previous][following]
    )

    # Cheapest insertion alone tends to consume near-depot customers early.
    # A modest radial reward gives more urgency to distant customers.
    depot_distance = matrix[depot][customer]

    # Reward a candidate when it leaves room for a geographically compatible
    # follow-up customer. Clarke-Wright saving measures the value of joining
    # their two depot spokes in one route.
    remaining = float(instance.capacity - instance.demands[customer])
    for node in route:
        remaining -= instance.demands[node]

    best_follow_saving = 0.0
    if remaining >= 0.0:
        for other in unassigned:
            if other == customer or instance.demands[other] > remaining:
                continue
            saving = (
                depot_distance
                + matrix[depot][other]
                - matrix[customer][other]
            )
            if saving > best_follow_saving:
                best_follow_saving = saving

    score = (
        insertion_cost
        - 0.50 * depot_distance
        - 0.25 * best_follow_saving
    )
    return float(score) if isfinite(score) else 0.0
