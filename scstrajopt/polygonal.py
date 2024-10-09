import numpy as np
from pydrake.all import ConvexSet, MathematicalProgram, eq
from pydrake.all import ClarabelSolver
from pybezier import BezierCurve, CompositeBezierCurve
from typing import List

def polygonal_curve(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[ConvexSet],
    vel_set: ConvexSet | None,
    acc_set: ConvexSet,
    deg: int,
    min_duration: float = 0
    ) -> CompositeBezierCurve:
    '''
    Constructs a polygonal composite curve that connects q_init and q_term in
    minimum time.
    '''

    # compute knots of polygonal curve and find kinks
    knots = get_knots(q_init, q_term, regions)
    kink_indices = get_kink_indices(knots)
    kinks = np.array([knots[i] for i in kink_indices])

    # turn polygonal curve into minimum-time bezier curve
    curves = []
    initial_time = 0
    point_to_point_solver = get_point_to_point_solver(vel_set, acc_set, deg)
    for kink, next_kink in zip(kinks[:-1], kinks[1:]):
        curve = point_to_point_solver(kink, next_kink, initial_time)
        initial_time = curve.final_time
        curves.append(curve)

    # split merged curves at knot points
    split_curves = []
    curve = curves.pop(0)
    for i, knot in enumerate(knots[1:]):
        if i + 1 in kink_indices:
            split_curves.append(curve)
            if curves:
                curve = curves.pop(0)
        else:
            split_time = get_crossing_time(curve, knot)
            split_curve, curve = curve.domain_split(split_time)
            split_curves.append(split_curve)

    return CompositeBezierCurve(split_curves)

def get_knots(q_init, q_term, regions):
    '''
    Computes the knot points of the shortest polygonal curve that connects the
    initial and final points through the given convex regions.
    '''

    # program and decision variables
    prog = MathematicalProgram()
    knots = prog.NewContinuousVariables(len(regions) + 1, len(q_init))
    slacks = prog.NewContinuousVariables(len(regions)) # slack variables for the L2 cost

    # cost function
    prog.AddLinearCost(sum(slacks))

    # constraints
    prog.AddLinearConstraint(eq(knots[0], q_init))
    prog.AddLinearConstraint(eq(knots[-1], q_term))
    for i, region in enumerate(regions):
        region.AddPointInSetConstraints(prog, knots[i])
        region.AddPointInSetConstraints(prog, knots[i + 1])
        distance = knots[i + 1] - knots[i]
        soc_vector = np.concatenate(([slacks[i]], distance))
        prog.AddLorentzConeConstraint(soc_vector)

    # solve program
    solver = ClarabelSolver()
    result = solver.Solve(prog)
    if not result.is_success():
        raise ValueError("Infeasible problem (could not find polygonal curve traversing the convex regions).")

    return result.GetSolution(knots)

def get_kink_indices(knots, tol=1e-4):
    '''
    Detects the indices of the points where the trajectory bends. Includes
    initial and final points. Uses the triangle inequality (division free):
    |knots[i] - knots[i-1]| + |knots[i+1] - knots[i]| > |knots[i+1] - knots[i-1]|
    implies that knots[i] is a kink.
    '''

    d01 = np.linalg.norm(knots[1:-1] - knots[:-2], axis=1)
    d12 = np.linalg.norm(knots[2:] - knots[1:-1], axis=1)
    d02 = np.linalg.norm(knots[2:] - knots[:-2], axis=1)
    residuals = d01 + d12 - d02
    internal_kink_indices = [i + 1 for i, res in enumerate(residuals) if res > tol]
    kink_indices = [0] + internal_kink_indices + [len(knots) - 1]

    return kink_indices

def get_crossing_time(curve, q, tol=1e-7):
    '''
    Uses bisection to find the time at which the curve (which is supposed to be
    a straigth line) goes through the point q.
    '''

    min_time = curve.initial_time
    max_time = curve.final_time
    q0 = curve.initial_point()
    direction = curve.final_point() - q0
    d1 = (q - q0).dot(direction)
    while max_time - min_time > tol:
        time = (min_time + max_time) / 2
        d2 = (curve(time) - q0).dot(direction)
        if d1 > d2:
            min_time = time
        else:
            max_time = time

    return (min_time + max_time) / 2

def get_point_to_point_solver(vel_set, acc_set, deg):
    '''
    Returns a program that optimizes a Bezier curve that moves between two
    points in minimum time. The initial and final velocities are set to zero.
    The problem would be nonconvex, but it becomes convex if we optimize over
    the square of the duration of the trajectory.
    '''

    # program and decision variables
    prog = MathematicalProgram()
    dim = acc_set.ambient_dimension()
    Q = prog.NewContinuousVariables(deg + 1, dim) # position control points
    V = prog.NewContinuousVariables(deg, dim) # velocity control points
    A = prog.NewContinuousVariables(deg - 1, dim) # acceleration control points
    T2 = prog.NewContinuousVariables(1)[0] # square of the curve duration

    # cost function
    prog.AddLinearCost(T2)

    # parametric constraints on initial and final position
    init_constr = prog.AddLinearConstraint(eq(Q[0], 0)).evaluator()
    term_constr = prog.AddLinearConstraint(eq(Q[-1], 0)).evaluator()
    
    # initial and final velocity constraints
    prog.AddLinearConstraint(eq(V[0], 0))
    prog.AddLinearConstraint(eq(V[-1], 0))

    # finite differences for control points of the derivatives
    diff = lambda X: (X[1:] - X[:-1]) * (len(X) - 1)
    prog.AddLinearConstraint(eq(V, diff(Q)))
    prog.AddLinearConstraint(eq(A, diff(V)))

    # velocity and acceleration constraints
    if vel_set is not None:
        T = prog.NewContinuousVariables(1)[0] # curve duration
        prog.AddRotatedLorentzConeConstraint(1, T2, T ** 2) # enforces T2 >= T^2
        for v in V:
            vel_set.AddPointInNonnegativeScalingConstraints(prog, v, T)
    for a in A:
        acc_set.AddPointInNonnegativeScalingConstraints(prog, a, T2)

    # instantiate solver
    solver = ClarabelSolver()

    def point_to_point_solver(q_init, q_term, initial_time):

        # update parametric constraints
        init_constr.UpdateCoefficients(init_constr.GetDenseA(), q_init)
        term_constr.UpdateCoefficients(init_constr.GetDenseA(), q_term)

        # solve program
        result = solver.Solve(prog)
        assert result.is_success()

        # reconstruct curve
        points = result.GetSolution(Q)
        duration = result.GetSolution(T2) ** .5
        final_time = initial_time + duration

        return BezierCurve(points, initial_time, final_time)

    return point_to_point_solver