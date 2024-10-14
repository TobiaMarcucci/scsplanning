import numpy as np
from scipy.optimize import brenth
from typing import List
from pydrake.all import ConvexSet, MathematicalProgram, eq
from pydrake.all import ClarabelSolver
from pybezier import BezierCurve, CompositeBezierCurve

def polygonal(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[ConvexSet],
    vel_set: ConvexSet | None,
    acc_set: ConvexSet,
    deg: int
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
    min_time_solver = get_min_time_solver(vel_set, acc_set, deg)
    for kink, next_kink in zip(kinks[:-1], kinks[1:]):
        curve = min_time_solver(kink, next_kink, initial_time)
        initial_time = curve.final_time
        curves.append(curve)

    # split merged curves at knot points
    split_curves = []
    curve = curves.pop(0)
    for i, knot in enumerate(knots[1:], start=1):
        if i in kink_indices:
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

def get_crossing_time(curve, q):
    '''
    Uses bisection to find the time at which the curve (which is supposed to be
    a straigth line) goes through the point q.
    '''

    distance = curve.final_point() - curve.initial_point()
    f = lambda t: (q - curve(t)).dot(distance)

    return brenth(f, curve.initial_time, curve.final_time)

def get_min_time_solver(vel_set, acc_set, deg):
    '''
    Returns a program that optimizes a Bezier curve that moves between two
    points in minimum time. The initial and final velocities are set to zero.
    The problem would be nonconvex, but it becomes convex if we optimize over
    the square of the duration of the trajectory.
    '''

    # program and decision variables
    prog = MathematicalProgram()
    Q = prog.NewContinuousVariables(deg + 1) # position control points
    V = prog.NewContinuousVariables(deg) # velocity control points
    A = prog.NewContinuousVariables(deg - 1) # acceleration control points
    T = prog.NewContinuousVariables(1)[0] # curve duration
    T2 = prog.NewContinuousVariables(1)[0] # square of the curve duration

    # cost function
    prog.AddLinearCost(T2)
    prog.AddRotatedLorentzConeConstraint(1, T2, T ** 2)

    # boundary constraints
    prog.AddLinearEqualityConstraint(Q[0] == 0)
    prog.AddLinearEqualityConstraint(V[0] == 0)
    prog.AddLinearEqualityConstraint(V[-1] == 0)

    # parametric constraints
    pos_term = prog.AddLinearEqualityConstraint(Q[-1] == 0).evaluator()

    # finite differences for computing the control points of the derivatives
    diff = lambda X: (X[1:] - X[:-1]) * (len(X) - 1)
    prog.AddLinearConstraint(eq(V, diff(Q)))
    prog.AddLinearConstraint(eq(A, diff(V)))

    # parametric velocity constraints
    M = np.zeros((deg, deg + 1)) # matrix
    np.fill_diagonal(M, 1)
    a = np.full(deg, - np.inf) # lower bound
    b = np.zeros(deg) # upper bound
    v = np.concatenate((V, [T])) # variables
    vel_ub = prog.AddLinearConstraint(M, a, b, v).evaluator()

    # parametric acceleration constraints
    M = M[:-1, :-1] # matrix
    a = a[:-1] # lower bound
    b = b[:-1] # upper bound
    v = np.concatenate((A, [T2])) # variables
    acc_ub = prog.AddLinearConstraint(M, a, b, v).evaluator()
    acc_lb = prog.AddLinearConstraint(M, b, - a, v).evaluator()

    # instantiate solver
    solver = ClarabelSolver()

    def min_time_solver(q_init, q_term, initial_time):

        # project problem onto the line connecting q_init and q_term
        direction = q_term - q_init
        distance = np.linalg.norm(direction)
        direction /= distance

        # update final-position constraint
        pos_term.UpdateCoefficients(Aeq=pos_term.GetDenseA(), beq=[distance])

        # get bounds on projected velocity and acceleration
        v_max = scale_vector(direction, vel_set)
        a_min = - scale_vector(- direction, acc_set)
        a_max = scale_vector(direction, acc_set)

        # update velocity and acceleration constraints
        def update_bound(constr, value):
            M = constr.GetDenseA()
            M[:, -1] = - value
            constr.UpdateCoefficients(M, constr.lower_bound(), constr.upper_bound())
        update_bound(vel_ub, v_max)
        update_bound(acc_lb, a_min)
        update_bound(acc_ub, a_max)

        # solve program
        result = solver.Solve(prog)

        # reconstruct curve
        points = q_init + np.outer(result.GetSolution(Q), direction)
        final_time = initial_time + result.GetSolution(T2) ** .5
        
        return BezierCurve(points, initial_time, final_time)

    return min_time_solver

def scale_vector(direction, set):
    '''
    Finds the maximum value of x such that x * direction lies in set.
    Assumes that set is convex and bounded.
    '''

    f = lambda s: 1 if set.PointInSet(direction * s) else -1
    s_max = 1
    while f(s_max) == 1:
        s_max *= 2
        
    return brenth(f, 0, s_max)