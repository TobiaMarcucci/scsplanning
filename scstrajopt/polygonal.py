import numpy as np
from pydrake.all import ConvexSet, MathematicalProgram, eq
from pydrake.all import ClarabelSolver
from pybezier import BezierCurve, CompositeBezierCurve
from typing import List

def polygonal(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[ConvexSet],
    vel_set: ConvexSet,
    acc_set: ConvexSet,
    deg: int,
    time_tol: float
    ) -> CompositeBezierCurve:
    '''
    Constructs a polygonal composite curve that connects q_init and q_term in
    minimum time.
    '''

    knots = get_knots(q_init, q_term, regions)
    curves = []
    initial_time = 0
    point_to_point_solver = get_point_to_point_solver(vel_set, acc_set, deg, time_tol)
    for knot, next_knot in zip(knots[:-1], knots[1:]):
        curve = point_to_point_solver(knot, next_knot, initial_time)
        initial_time = curve.final_time
        curves.append(curve)

    return CompositeBezierCurve(curves)

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

def get_point_to_point_solver(vel_set, acc_set, deg, time_tol):
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
    T = prog.NewContinuousVariables(1)[0] # curve duration
    T2 = prog.NewContinuousVariables(1)[0] # square of the curve duration
    
    # cost function
    prog.AddLinearCost(T2)
    prog.AddLinearConstraint(T2 >= time_tol ** 2)
    prog.AddRotatedLorentzConeConstraint(1, T2, T ** 2) # enforces T2 >= T^2

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