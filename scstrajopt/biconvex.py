import numpy as np
from typing import List
from pydrake.all import ConvexSet, MathematicalProgram, ClarabelSolver
from pybezier import BezierCurve, CompositeBezierCurve
from scstrajopt.fixed_points import FixedPoints
from scstrajopt.fixed_velocities import FixedVelocities

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
    fixed_position = FixedPoints(regions, vel_set, acc_set, deg, relaxation=True)
    curve = polygonal_curve(q_init, q_term, regions)
    curve = fixed_position.solve(curve)

    # instantiate programs for the biconvex alternation
    fixed_position = FixedPoints(regions, vel_set, acc_set, deg)
    fixed_velocity = FixedVelocities(q_init, q_term, regions, vel_set, acc_set, deg)

    # alternate until curve duration does not decrease sufficiently
    rel_improvement = lambda duration, curve: (duration - curve.duration) / curve.duration
    velocity_duration = np.inf
    position_duration = curve.duration

    while True:

        # fixed transition velocities
        curve = fixed_velocity.solve(curve)
        if rel_improvement(velocity_duration, curve) < tol:
            break
        velocity_duration = curve.duration

        # fixed transition points 
        curve = fixed_position.solve(curve)
        if rel_improvement(position_duration, curve) < tol:
            break
        position_duration = curve.duration

    return curve

def polygonal_curve(q_init, q_term, regions):
    '''
    Computes the knot points of the shortest polygonal curve that connects the
    initial and final points through the given convex regions.
    '''

    # problem size
    reg = len(regions)
    dim = len(q_init)

    # program and decision variables
    prog = MathematicalProgram()
    points = prog.NewContinuousVariables(reg + 1, dim)
    slacks = prog.NewContinuousVariables(reg) # slack variables for the L2 cost

    # cost function
    prog.AddLinearCost([1] * reg, 0, slacks)

    # initial conditions
    I = np.eye(dim)
    prog.AddLinearEqualityConstraint(I, q_init, points[0])
    prog.AddLinearEqualityConstraint(I, q_term, points[-1])

    # points in convex regions
    for i, region in enumerate(regions):
        region.AddPointInSetConstraints(prog, points[i])
        region.AddPointInSetConstraints(prog, points[i + 1])

    # cost slacks
    D = np.zeros((dim + 1, 2 * dim + 1))
    D[0, 0] = 1
    D[1:, 1:dim+1] = I
    D[1:, dim+1:] = - I
    d = np.zeros(dim + 1)
    for i in range(reg):
        vars = np.concatenate(([slacks[i]], points[i], points[i + 1]))
        prog.AddLorentzConeConstraint(D, d, vars)

    # solve program
    solver = ClarabelSolver()
    result = solver.Solve(prog)
    if not result.is_success():
        raise ValueError("Infeasible problem (could not find polygonal curve traversing the convex regions).")

    # construct a fake curve
    points = result.GetSolution(points)
    curves = []
    for i in range(reg):
        curves.append(BezierCurve(points[i:i+2], i, i+1))

    return CompositeBezierCurve(curves)
