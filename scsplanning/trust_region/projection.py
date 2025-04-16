import numpy as np
import pydrake as pd
from pybezier import BezierCurve, CompositeBezierCurve

def projection(q_init, q_term, regions, vel_set, acc_set, deg):

    # problem size
    reg = len(regions)
    dim = len(q_init)

    # program and decision variables
    prog = pd.MathematicalProgram()
    Q = [prog.NewContinuousVariables(deg + 1, dim) for i in range(reg)] # control points of configuration
    Qdot = [prog.NewContinuousVariables(deg, dim) for i in range(reg)] # control points of velocity
    Rdot = [prog.NewContinuousVariables(deg, dim) for i in range(reg)] # control points of velocity
    Qddot = [prog.NewContinuousVariables(deg - 1, dim) for i in range(reg)] # control points of acceleration
    Rddot = [prog.NewContinuousVariables(deg - 1, dim) for i in range(reg)] # control points of acceleration

    # boundary conditions
    I = np.eye(dim)
    zero = [0] * dim
    prog.AddLinearEqualityConstraint(I, q_init, Q[0][0])
    prog.AddLinearEqualityConstraint(I, q_term, Q[-1][-1])
    prog.AddLinearEqualityConstraint(I, zero, Rdot[0][0])
    prog.AddLinearEqualityConstraint(I, zero, Rdot[-1][-1])

    # continuity of position and velocity
    D = np.hstack((I, -I))
    for i in range(1, reg):
        vars = np.concatenate((Q[i-1][-1], Q[i][0]))
        prog.AddLinearEqualityConstraint(D, zero, vars)
        vars = np.concatenate((Rdot[i-1][-1], Rdot[i][0]))
        prog.AddLinearEqualityConstraint(D, zero, vars)

    # finite difference for velocity
    D = np.hstack((I, -I*deg, I*deg))
    for i in range(reg):
        for k in range(deg):
            vars = np.concatenate((Qdot[i][k], Q[i][k+1], Q[i][k]))
            prog.AddLinearEqualityConstraint(D, zero, vars)

    # finite difference for acceleration
    D = np.hstack((I, -I*(deg-1), I*(deg-1)))
    for i in range(reg):
        for k in range(deg - 1):
            vars = np.concatenate((Qddot[i][k], Rdot[i][k+1], Rdot[i][k]))
            prog.AddLinearEqualityConstraint(D, zero, vars)

    # convex constraints on position
    for i, region in enumerate(regions):
        if i == 0:
            Qi = Q[i][1:]
        if i == reg - 1:
            Qi = Q[i][:-1]
        else:
            Qi = Q[i]
        for q in Qi:
            region.AddPointInSetConstraints(prog, q)

    # convex constraints on velocity
        if i == 0:
            Vi = Rdot[i][1:-1]
        else:
            Vi = Rdot[i][:-1]
        for v in Vi:
            vel_set.AddPointInSetConstraints(prog, v)

    # convex constraints on acceleration
    for Ai, Ti in zip(Rddot, T):
        for a in Ai:
            acc_set.AddPointInSetConstraints(prog, a, Ti)

    # linearized bilinear constraints
    D = np.zeros((dim, 2 * dim))
    d = np.zeros(dim)
    linearized_vel = []
    linearized_acc = []
    for i in range(reg):
        linearized_vel_i = []
        linearized_acc_i = []
        for k in range(deg):
            vars = np.concatenate((Qdot[i][k], Rdot[i][k]))
            linearized_vel_i.append(prog.AddLinearEqualityConstraint(D, d, vars))
        for k in range(deg - 1):
            vars = np.concatenate((Qddot[i][k], Rddot[i][k]))
            linearized_acc_i.append(prog.AddLinearEqualityConstraint(D, d, vars))
        linearized_vel.append(linearized_vel_i)
        linearized_acc.append(linearized_acc_i)

    def projection_solver(T_nom):

        # update linearized bilinear constraints
        for i, linearized_vel_i in enumerate(linearized_vel):
            for linearized_vel_ik in linearized_vel_i:
                D = np.hstack((I, - T_nom[i] * I))
                linearized_vel_ik.UpdateCoefficients(D, zero)
        for i, linearized_acc_i in enumerate(linearized_acc):
            for linearized_acc_ik in linearized_acc_i:
                D = np.hstack((I, - T_nom[i] * I))
                linearized_acc_ik.UpdateCoefficients(D, zero)

        # solve program
        solver = pd.ClarabelSolver()
        result = solver.Solve(prog)
        if not result.is_success():
            return None

        # reconstruct Bezier curve from control points
        beziers = []
        dT = np.cumsum([0] + list(T_nom))
        beziers = [BezierCurve(result.GetSolution(Q[i]), dT[i], dT[i+1]) for i in range(reg)]
        return CompositeBezierCurve(beziers)
    
    return projection_solver
