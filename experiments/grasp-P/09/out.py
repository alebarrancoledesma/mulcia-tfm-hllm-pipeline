from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cvrp import Instance


def _clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


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

    insertion_delta = (
        distances[predecessor][customer]
        + distances[customer][successor]
        - distances[predecessor][successor]
    )

    # A maximum radial distance gives a stable geometric normalization and is
    # shared by every candidate evaluated for this construction state.
    scale = 0.0
    route_load = 0.0
    for node in route:
        route_load += float(demands[node])
        radial = distances[depot][node]
        if radial > scale:
            scale = radial

    for node in unassigned:
        if node != depot:
            radial = distances[depot][node]
            if radial > scale:
                scale = radial

    if not isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0

    demand = float(demands[customer])
    load_after = route_load + demand
    remaining_after = capacity - load_after
    if remaining_after < 0.0:
        remaining_after = 0.0
    elif remaining_after > capacity:
        remaining_after = capacity

    stage = _clamp(load_after / capacity, 0.0, 1.0)
    residual_fraction = _clamp(remaining_after / capacity, 0.0, 1.0)

    # Inspect one possible continuation. It jointly favors a nearby next
    # customer and a next demand that would leave little residual capacity.
    best_continuation: float | None = None
    tolerance = 1.0e-12 * max(1.0, capacity)
    for other in unassigned:
        if other == customer or other == depot:
            continue
        other_demand = float(demands[other])
        if other_demand <= remaining_after + tolerance:
            spatial = distances[customer][other] / scale
            spatial = _clamp(spatial, 0.0, 2.0)
            gap = _clamp(
                (remaining_after - other_demand) / capacity,
                0.0,
                1.0,
            )
            continuation = 0.55 * spatial + gap
            if best_continuation is None or continuation < best_continuation:
                best_continuation = continuation

    if best_continuation is None:
        # If insertion would close the route, unused capacity is the relevant
        # penalty; a nearly full route is therefore inexpensive to close.
        continuation_penalty = 1.2 * residual_fraction
    else:
        continuation_penalty = best_continuation

    radial_fraction = _clamp(distances[depot][customer] / scale, 0.0, 1.0)

    # Cheapest insertion supplies local tour quality. A modest radial-savings
    # reward prevents centrally located customers from systematically taking
    # capacity needed by remote customers. Capacity lookahead becomes stronger
    # as the route fills.
    radial_weight = 0.14 + 0.08 * (1.0 - stage)
    continuation_weight = 0.10 + 0.25 * stage
    score = (
        insertion_delta / scale
        - radial_weight * radial_fraction
        + continuation_weight * continuation_penalty
    )

    if not isfinite(score):
        return 0.0
    return float(_clamp(score, -1.0e100, 1.0e100))
