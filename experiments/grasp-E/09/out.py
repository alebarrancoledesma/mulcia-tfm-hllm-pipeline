from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def _candidate_two_opt_gain(
    distance_matrix: list[list[float]],
    depot: int,
    route: list[int],
    customer: int,
    position: int,
) -> float:
    """Best one-step 2-opt gain involving an edge incident to customer."""
    nodes = [depot]
    nodes.extend(route[:position])
    nodes.append(customer)
    nodes.extend(route[position:])
    nodes.append(depot)

    last_edge = len(nodes) - 2
    if last_edge < 2:
        return 0.0

    # The inserted customer's incoming and outgoing edge indices.
    candidate_edges = (position, position + 1)
    best_gain = 0.0

    for candidate_edge in candidate_edges:
        for other_edge in range(last_edge + 1):
            i = candidate_edge
            j = other_edge
            if i == j:
                continue
            if i > j:
                i, j = j, i

            # Adjacent edges cannot define a proper 2-opt reversal. The first
            # and last edges are also adjacent through the depot.
            if j == i + 1 or (i == 0 and j == last_edge):
                continue

            a = nodes[i]
            b = nodes[i + 1]
            u = nodes[j]
            v = nodes[j + 1]
            gain = (
                distance_matrix[a][u]
                + distance_matrix[b][v]
                - distance_matrix[a][b]
                - distance_matrix[u][v]
            )
            if gain < best_gain:
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
    depot = instance.depot
    distances = instance.distance_matrix

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    # Credit part of the best 2-opt repair directly enabled by the insertion.
    repair_gain = _candidate_two_opt_gain(
        distances, depot, route, customer, position
    )

    route_load = 0.0
    for node in route:
        route_load += instance.demands[node]

    residual = (
        float(instance.capacity)
        - route_load
        - float(instance.demands[customer])
    )
    if residual < 0.0:
        residual = 0.0

    customer_radius = distances[depot][customer]
    radial_sum = customer_radius
    radial_count = 1
    best_pair_saving = 0.0
    largest_fitting_demand = 0.0
    has_fitting_successor = False

    for other in unassigned:
        if other == customer:
            continue

        other_radius = distances[depot][other]
        radial_sum += other_radius
        radial_count += 1

        other_demand = float(instance.demands[other])
        if other_demand <= residual:
            has_fitting_successor = True
            if other_demand > largest_fitting_demand:
                largest_fitting_demand = other_demand

            # Saving from serving customer and other in a connected spatial
            # cluster rather than through independent depot spokes.
            pair_saving = (
                customer_radius
                + other_radius
                - distances[customer][other]
            )
            if pair_saving > best_pair_saving:
                best_pair_saving = pair_saving

    distance_scale = radial_sum / radial_count
    if distance_scale < 1.0e-12:
        distance_scale = 1.0e-12

    if has_fitting_successor:
        packing_gap = residual - largest_fitting_demand
    else:
        packing_gap = residual
    if packing_gap < 0.0:
        packing_gap = 0.0

    packing_penalty = (
        0.06
        * distance_scale
        * packing_gap
        / float(instance.capacity)
    )

    score = (
        insertion_delta
        + 0.75 * repair_gain
        - 0.62 * customer_radius
        - 0.14 * best_pair_saving
        + packing_penalty
    )

    # Keep the API promise even for numerically pathological input data.
    if score != score:
        return 1.0e300
    if score > 1.0e300:
        return 1.0e300
    if score < -1.0e300:
        return -1.0e300
    return float(score)
