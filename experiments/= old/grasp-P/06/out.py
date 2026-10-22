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
    """Return the net merge cost of inserting a customer into this route."""
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
    standalone_cost = 2.0 * matrix[depot][customer]

    return float(insertion_cost - standalone_cost)
