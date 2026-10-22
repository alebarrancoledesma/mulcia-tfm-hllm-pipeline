from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def _path_vertex(
    route: list[int],
    customer: int,
    position: int,
    path_index: int,
    depot: int,
) -> int:
    """Return a vertex of depot-route-depot after conceptual insertion."""
    route_length = len(route)
    if path_index == 0 or path_index == route_length + 2:
        return depot

    sequence_index = path_index - 1
    if sequence_index == position:
        return customer
    if sequence_index < position:
        return route[sequence_index]
    return route[sequence_index - 1]


def heuristic(
    instance: Instance,
    solution: list[list[int]],
    unassigned: list[int],
    route_index: int,
    customer: int,
    position: int,
) -> float:
    """Score insertion by corrected singleton saving; lower is better."""
    route = solution[route_index]
    matrix = instance.distance_matrix
    depot = instance.depot
    route_length = len(route)

    previous = depot if position == 0 else route[position - 1]
    following = depot if position == route_length else route[position]
    insertion_cost = (
        matrix[previous][customer]
        + matrix[customer][following]
        - matrix[previous][following]
    )

    # Estimate what the final 2-opt phase can immediately recover. Only moves
    # involving one of the two newly created edges need to be scanned, keeping
    # the evaluation linear rather than examining every pair of route edges.
    edge_count = route_length + 2
    inserted_edges = (position, position + 1)
    best_gain = 0.0

    for inserted_edge in inserted_edges:
        for other_edge in range(edge_count):
            left = min(inserted_edge, other_edge)
            right = max(inserted_edge, other_edge)

            # Equal or adjacent edges do not define a proper 2-opt move.
            if right - left <= 1:
                continue
            # The first and last edges share the depot in the route cycle.
            if left == 0 and right == edge_count - 1:
                continue

            a = _path_vertex(route, customer, position, left, depot)
            b = _path_vertex(route, customer, position, left + 1, depot)
            c = _path_vertex(route, customer, position, right, depot)
            d = _path_vertex(route, customer, position, right + 1, depot)

            gain = (
                matrix[a][b]
                + matrix[c][d]
                - matrix[a][c]
                - matrix[b][d]
            )
            if gain > best_gain:
                best_gain = gain

    corrected_cost = insertion_cost - best_gain

    # A singleton route for this customer costs twice its depot distance.
    # Subtracting that baseline ranks candidates by the distance saved by
    # sharing the current route, while naturally retaining far, compatible
    # customers near the farthest-first seed.
    singleton_cost = 2.0 * matrix[depot][customer]
    return float(corrected_cost - singleton_cost)
