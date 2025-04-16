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
    vel_set: ConvexSet,
    acc_set: ConvexSet,
    deg: int
    ) -> CompositeBezierCurve:
    '''
    Constructs a polygonal composite curve that connects q_init and q_term in minimum time.
    '''

    # compute points of polygonal curve and find vertices
    points = get_points(q_init, q_term, regions)
    vertex_indices = get_vertex_indices(points)
    vertices = [points[i] for i in vertex_indices]

    # turn polygonal curve into minimum-time bezier curve
    curves = []
    initial_time = 0
    min_time_solver = get_min_time_solver(vel_set, acc_set, deg)
    for vertex, next_vertex in zip(vertices[:-1], vertices[1:]):
        curve = min_time_solver(vertex, next_vertex, initial_time)
        initial_time = curve.final_time
        curves.append(curve)

    # split merged curves at knot points
    split_curves = []
    curve = curves.pop(0)
    for i, knot in enumerate(points[1:], start=1):
        if i in vertex_indices:
            split_curves.append(curve)
            if curves:
                curve = curves.pop(0)
        else:
            split_time = get_crossing_time(curve, knot)
            split_curve, curve = curve.domain_split(split_time)
            split_curves.append(split_curve)

    return CompositeBezierCurve(split_curves)

def get_points(q_init, q_term, regions):
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

    return result.GetSolution(points)

def get_vertex_indices(points, tol=1e-4):
    '''
    Detects the indices of the points where the trajectory bends. Includes
    initial and final points. Uses the triangle inequality (division free):
    |points[i] - points[i-1]| + |points[i+1] - points[i]| > |points[i+1] - points[i-1]|
    implies that points[i] is a vertex.
    This assumes that consecutive vertices are not equal:
    points[i] != points[i+1] for all i.
    '''

    d01 = np.linalg.norm(points[1:-1] - points[:-2], axis=1)
    d12 = np.linalg.norm(points[2:] - points[1:-1], axis=1)
    d02 = np.linalg.norm(points[2:] - points[:-2], axis=1)
    residuals = d01 + d12 - d02
    vertex_indices = [i + 1 for i, r in enumerate(residuals) if r > tol]

    return [0] + vertex_indices + [len(points) - 1]

def get_crossing_time(curve, q):
    '''
    Uses bisection to find the time at which the curve (which is supposed to be
    a straigth line) goes through the point q.
    '''

    distance = curve.final_point - curve.initial_point
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
    S = prog.NewContinuousVariables(1)[0] # inverse of the curve duration

    # constraint time and inverse time
    D = [[1, 0], [0, 1], [0, 0]]
    d = [0, 0, 1]
    prog.AddRotatedLorentzConeConstraint(D, d, [T, S])

    # cost function
    D = [1]
    d = 0
    prog.AddLinearCost(D, d, [T])
    
    # boundary constraints
    prog.AddLinearEqualityConstraint(D, d, [Q[0]])
    prog.AddLinearEqualityConstraint(D, d, [V[0]])
    prog.AddLinearEqualityConstraint(D, d, [V[-1]])

    # parametric final position
    D = [1, 0]
    pos_term = prog.AddLinearEqualityConstraint(D, d, [Q[-1], S]).evaluator()

    # finite differences for velocity
    I = np.eye(deg)
    D = np.hstack((I, -I*deg, I*deg))
    d = np.zeros(deg)
    vars = np.concatenate((V, Q[1:], Q[:-1]))
    prog.AddLinearEqualityConstraint(D, d, vars)

    # finite differences for acceleration
    I = I[1:, 1:]
    D = np.hstack((I, -I*(deg-1), I*(deg-1)))
    d = d[1:]
    vars = np.concatenate((A, V[1:], V[:-1]))
    prog.AddLinearEqualityConstraint(D, d, vars)

    # parametric velocity constraints
    D = np.eye(deg)
    d = np.full(deg, np.inf)
    vel_ub = prog.AddLinearConstraint(D, -d, d, V).evaluator()

    # parametric acceleration constraints
    D = D[:-1, :]
    d = d[1:]
    e = np.zeros(deg - 1)
    vars = np.concatenate((A, [T]))
    acc_lb = prog.AddLinearConstraint(D, e, d, vars).evaluator()
    acc_ub = prog.AddLinearConstraint(D, -d, e, vars).evaluator()

    # instantiate solver
    solver = ClarabelSolver()

    def min_time_solver(q_init, q_term, initial_time):

        # project problem onto the line connecting q_init and q_term
        direction = q_term - q_init
        distance = np.linalg.norm(direction)
        direction /= distance

        # update final-position constraint
        D = [[1, -distance]]
        pos_term.UpdateCoefficients(D, [0])

        # update velocity upper bound
        d = vel_ub.upper_bound()
        d[:] = scale_vector(direction, vel_set)
        vel_ub.UpdateCoefficients(vel_ub.GetDenseA(), vel_ub.lower_bound(), d)

        # update acceleration lower bound
        D = acc_lb.GetDenseA()
        D[:, -1] = scale_vector(- direction, acc_set)
        acc_lb.UpdateCoefficients(D, acc_lb.lower_bound(), acc_lb.upper_bound())

        # update acceleration upper bound
        D[:, -1] = - scale_vector(direction, acc_set)
        acc_ub.UpdateCoefficients(D, acc_ub.lower_bound(), acc_ub.upper_bound())

        # solve program
        result = solver.Solve(prog)
        assert result.is_success()

        # reconstruct curve
        T_opt = result.GetSolution(T)
        Q_opt = result.GetSolution(Q) * T_opt
        points = q_init + np.outer(Q_opt, direction)
        final_time = initial_time + T_opt
        
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
