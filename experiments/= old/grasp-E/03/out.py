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
    """Score an insertion by its cost after one candidate-touching 2-opt move."""
    route = solution[route_index]
    matrix = instance.distance_matrix
    depot = instance.depot
    route_size = len(route)

    previous = depot if position == 0 else route[position - 1]
    following = depot if position == route_size else route[position]
    insertion_cost = (
        matrix[previous][customer]
        + matrix[customer][following]
        - matrix[previous][following]
    )

    # Number of customers after the hypothetical insertion. Edge indices in
    # the depot-to-depot representation then range from 0 through new_size.
    new_size = route_size + 1

    def vertex(index: int) -> int:
        """Return vertex at an index of the hypothetical depot-to-depot path."""
        if index == 0 or index == new_size + 1:
            return depot
        route_position = index - 1
        if route_position == position:
            return customer
        if route_position < position:
            return route[route_position]
        return route[route_position - 1]

    # Examine all valid 2-opt moves that remove either edge incident to the
    # inserted customer. This is linear rather than a full quadratic 2-opt.
    best_gain = 0.0
    for candidate_edge in (position, position + 1):
        for other_edge in range(new_size + 1):
            left = candidate_edge
            right = other_edge
            if left > right:
                left, right = right, left

            # Equal or adjacent edges do not define a proper 2-opt move. The
            # first and last edges are also adjacent through the depot.
            if left == right or right == left + 1:
                continue
            if left == 0 and right == new_size:
                continue

            a = vertex(left)
            b = vertex(left + 1)
            c = vertex(right)
            d = vertex(right + 1)
            gain = (
                matrix[a][b]
                + matrix[c][d]
                - matrix[a][c]
                - matrix[b][d]
            )
            if gain > best_gain:
                best_gain = gain

    # A modest generalized-insertion depot credit discourages leaving remote
    # customers for separate later routes. Larger demands receive slightly
    # more credit because they are harder to accommodate late in construction.
    demand_fraction = float(instance.demands[customer]) / float(instance.capacity)
    if demand_fraction < 0.0:
        demand_fraction = 0.0
    elif demand_fraction > 1.0:
        demand_fraction = 1.0
    radial_weight = 0.35 + 0.25 * demand_fraction
    depot_credit = radial_weight * matrix[depot][customer]

    return float(insertion_cost - best_gain - depot_credit)
