import numpy as np
from typing import List
from pydrake.all import ConvexSet
from pybezier import CompositeBezierCurve
from scsplanning import polygonal
from scsplanning.fixed_points import FixedPoints
from scsplanning.fixed_velocities import FixedVelocities

def biconvex(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[ConvexSet],
    vel_set: ConvexSet,
    acc_set: ConvexSet,
    deg: int,
    tol: float = 1e-2
    ) -> CompositeBezierCurve:

    # compute initial guess
    curve = polygonal(q_init, q_term, regions, vel_set, acc_set, deg)

    # instantiate programs for the biconvex alternation
    point_program = FixedPoints(regions, vel_set, acc_set, deg)
    velocity_program = FixedVelocities(q_init, q_term, regions, vel_set, acc_set, deg)

    # relative improvement between iterations
    improvement = lambda duration, curve: (duration - curve.duration) / curve.duration
    point_duration = np.inf
    velocity_duration = curve.duration

    # alternate until curve duration does not decrease sufficiently
    while True:

        # fixed transition points 
        curve = point_program.solve(curve)
        if improvement(point_duration, curve) < tol:
            break
        point_duration = curve.duration

        # fixed transition velocities 
        curve = velocity_program.solve(curve)
        if improvement(velocity_duration, curve) < tol:
            break
        velocity_duration = curve.duration

    return curve
