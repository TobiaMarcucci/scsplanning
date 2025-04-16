import numpy as np
from typing import List
from pydrake.all import ConvexSet
from pybezier import CompositeBezierCurve
from scsplanning import polygonal
from scsplanning.trust_region.tangent import tangent
from scsplanning.trust_region.projection import projection


def biconvex(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[ConvexSet],
    vel_set: ConvexSet,
    acc_set: ConvexSet,
    deg: int,
    kappa: float = 1,
    omega: float = 3,
    tol: float = 1e-2
    ) -> CompositeBezierCurve:

    # compute initial guess
    curve = polygonal(q_init, q_term, regions, vel_set, acc_set, deg)

    # instantiate programs for the biconvex alternation
    tangent_solver = tangent(q_init, q_term, regions, vel_set, acc_set, deg)
    projection_solver = projection(q_init, q_term, regions, vel_set, acc_set, deg)

    # alternate until curve duration does not decrease sufficiently
    while True:

        # fixed transition points 
        traversal_times = tangent_solver(curve, kappa)

        # check convergence
        if (curve.duration - sum(traversal_times)) / curve.duration < tol:
             break

        # update trust region
        ratios = np.divide(traversal_times, curve.durations())
        kappa_max = max(max(ratios), 1 / min(ratios)) - 1
        kappa = kappa_max / omega

        # update curve
        new_curve = projection_solver(traversal_times)
        if new_curve is not None:
            curve = new_curve

    return curve

        
