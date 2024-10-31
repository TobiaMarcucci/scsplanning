import numpy as np
from typing import List
from pydrake.all import ConvexSet, MathematicalProgram, eq, ge, le
from pydrake.all import SnoptSolver, IpoptSolver
from pybezier import BezierCurve, CompositeBezierCurve
from scstrajopt import polygonal
import pydrake.solvers as mp


def nonconvex(q_init, q_term, regions, vel_set, acc_set, deg, region_tol=1e-3, solver=IpoptSolver):
    '''
    Solves the motion planning problem by calling a solver for nonconvex
    optimization.
    '''

    # problem size
    reg = len(regions)
    dim = len(q_init)

    # program and decision variables
    prog = MathematicalProgram()
    T = prog.NewContinuousVariables(reg) # time durations of trajectory pieces
    T2 = prog.NewContinuousVariables(reg) # square of time durations
    Q = [prog.NewContinuousVariables(deg + 1, dim) for i in range(reg)] # control points of configuration
    V = [prog.NewContinuousVariables(deg, dim) for i in range(reg)] # control points of velocity
    A = [prog.NewContinuousVariables(deg - 1, dim) for i in range(reg)] # control points of acceleration

    # link between time variables
    D = [[2, 0], [0, 0]]
    d = [0, -1]
    hessian_type = mp.QuadraticConstraint.HessianType.kPositiveSemidefinite
    for vars in zip(T, T2):
        prog.AddQuadraticConstraint(D, d, 0, 0, vars, hessian_type=hessian_type)

    # boundary conditions
    I = np.eye(dim)
    zero = [0] * dim
    prog.AddLinearEqualityConstraint(I, q_init, Q[0][0])
    prog.AddLinearEqualityConstraint(I, q_term, Q[-1][-1])
    prog.AddLinearEqualityConstraint(I, zero, V[0][0])
    prog.AddLinearEqualityConstraint(I, zero, V[-1][-1])

    # continuity of position
    D = np.hstack((I, -I))
    for i in range(1, reg):
        vars = np.concatenate((Q[i-1][-1], Q[i][0]))
        prog.AddLinearEqualityConstraint(D, zero, vars)

    # continuity of velocity
    D = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]]
    d = [0] * 4
    hessian_type = mp.QuadraticConstraint.HessianType.kIndefinite
    for i in range(1, reg):
        for k in range(dim):
            vars = np.array([V[i-1][-1, k], T[i], V[i][0, k], T[i - 1]])
            prog.AddQuadraticConstraint(D, d, 0, 0, vars, hessian_type=hessian_type)

    # finite difference for velocity
    D = np.hstack((I, -I*deg, I*deg))
    for i in range(reg):
        for k in range(deg):
            vars = np.concatenate((V[i][k], Q[i][k+1], Q[i][k]))
            prog.AddLinearEqualityConstraint(D, zero, vars)

    # finite difference for acceleration
    D = np.hstack((I, -I*(deg-1), I*(deg-1)))
    for i in range(reg):
        for k in range(deg - 1):
            vars = np.concatenate((A[i][k], V[i][k+1], V[i][k]))
            prog.AddLinearEqualityConstraint(D, zero, vars)

    # convex constraints on control points
    for i in range(reg):
        for q in Q[i]:
            regions[i].AddPointInSetConstraints(prog, q) 
        for v in V[i]:
            vel_set.AddPointInNonnegativeScalingConstraints(prog, v, T[i])
        for a in A[i]:
            acc_set.AddPointInNonnegativeScalingConstraints(prog, a, T2[i])

    # cost function
    prog.AddLinearCost([1] * reg, 0, T)

    # warm start with minimum-time polygonal curve
    curve = polygonal(q_init, q_term, regions, vel_set, acc_set, deg)
    T_guess = np.array([bez.duration for bez in curve])
    prog.SetInitialGuess(T, T_guess)
    prog.SetInitialGuess(T2, T_guess ** 2)
    velocity = curve.derivative()
    acceleration = velocity.derivative()
    for i in range(reg):
        prog.SetInitialGuess(Q[i], curve[i].points)
        prog.SetInitialGuess(V[i], T_guess[i] * velocity[i].points)
        prog.SetInitialGuess(A[i], T_guess[i] ** 2 * acceleration[i].points)

    # solve program
    result = solver().Solve(prog)
    if not result.is_success():
        return None

    # reconstruct Bezier curve from control points
    beziers = []
    time_start = 0
    for i in range(reg):
        Ti = result.GetSolution(T[i])
        if Ti >= region_tol:
            time_stop = time_start + max(Ti, 0)
            bez = BezierCurve(result.GetSolution(Q[i]), time_start, time_stop)
            beziers.append(bez)
        time_start = time_stop

    return CompositeBezierCurve(beziers)
