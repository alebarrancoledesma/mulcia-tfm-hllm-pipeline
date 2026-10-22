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

    predecessor = depot if position == 0 else route[position - 1]
    successor = depot if position == len(route) else route[position]

    # Exact increase in the current closed tour at this insertion position.
    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    load = 0.0
    radial_sum = 0.0
    radial_count = 0
    for node in route:
        load += float(demands[node])
        radial_sum += distances[depot][node]
        radial_count += 1

    customer_demand = float(demands[customer])
    residual = capacity - load - customer_demand
    if residual < 0.0:
        residual = 0.0

    # Look for a capacity-compatible nearby successor. Its saving measures how
    # much cheaper it is to reach that customer from this candidate than from
    # the depot. Also collect a candidate-independent local distance scale.
    best_successor_saving = 0.0
    has_fitting_successor = False
    for other in unassigned:
        if other == customer or other == depot:
            continue

        radial = distances[depot][other]
        radial_sum += radial
        radial_count += 1

        if float(demands[other]) <= residual:
            has_fitting_successor = True
            saving = radial - distances[customer][other]
            if saving > best_successor_saving:
                best_successor_saving = saving

    if radial_count:
        scale = radial_sum / radial_count
    else:
        scale = distances[depot][customer]
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0

    if capacity > 0.0:
        fill_before = min(1.0, max(0.0, load / capacity))
        demand_fraction = min(1.0, max(0.0, customer_demand / capacity))
        residual_fraction = min(1.0, max(0.0, residual / capacity))
    else:
        fill_before = 0.0
        demand_fraction = 0.0
        residual_fraction = 0.0

    # Radial credit implements a moderated route-saving effect and fades as
    # the vehicle fills, allowing later insertions to turn back toward depot.
    radial_credit = (0.55 - 0.25 * fill_before) * distances[depot][customer]

    # Large demands are mildly urgent, especially late in construction, since
    # delaying them tends to make them incompatible with partially filled routes.
    demand_credit = scale * (0.18 + 0.22 * fill_before) * demand_fraction

    # Prefer candidates that leave a geographically coherent feasible follow-up.
    continuation_credit = 0.12 * best_successor_saving

    # If this insertion necessarily closes the route, distinguish tight packing
    # from closures that strand substantial unused capacity.
    closure_penalty = 0.0
    if not has_fitting_successor:
        closure_penalty = 0.45 * scale * residual_fraction

    score = (
        insertion_delta
        - radial_credit
        - demand_credit
        - continuation_credit
        + closure_penalty
    )

    if not math.isfinite(score):
        return 1.0e300
    return float(max(-1.0e300, min(1.0e300, score)))
