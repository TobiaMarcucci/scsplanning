import numpy as np
import pydrake as pd

def tangent(q_init, q_term, regions, vel_set, acc_set, deg):

    # problem size
    reg = len(regions)
    dim = len(q_init)

    # program and decision variables
    prog = pd.MathematicalProgram()
    T = prog.NewContinuousVariables(reg) # time durations of trajectory pieces
    Q = [prog.NewContinuousVariables(deg + 1, dim) for i in range(reg)] # control points of configuration
    Qdot = [prog.NewContinuousVariables(deg, dim) for i in range(reg)] # control points of velocity
    Rdot = [prog.NewContinuousVariables(deg, dim) for i in range(reg)] # control points of velocity
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
            vars = np.concatenate((Rddot[i][k], Rdot[i][k+1], Rdot[i][k]))
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
            acc_set.AddPointInNonnegativeScalingConstraints(prog, a, Ti)

    # cost function
    prog.AddLinearCost([1] * reg, 0, T)

    # linearized bilinear constraints
    D = np.zeros((dim, 2 * dim + 1))
    d = np.zeros(dim)
    linearized = []
    for i in range(reg):
        linearized_i = []
        for k in range(deg):
            vars = np.concatenate((Qdot[i][k], Rdot[i][k], [T[i]]))
            linearized_i.append(prog.AddLinearEqualityConstraint(D, d, vars))
        linearized.append(linearized_i)

    # trust-region constraints
    D = np.eye(reg)
    d = np.zeros(reg)
    trust_region = prog.AddLinearConstraint(D, d, d, T)

    def tangent_solver(curve, kappa):

        # extract relevant data from curve
        T_nom = curve.durations()
        Rdot_nom = [c.points for c in curve.derivative()]

        # update linearized bilinear constraints
        for i, linearized_i in enumerate(linearized):
            for k, linearized_ik in enumerate(linearized_i):
                D = np.vstack((I, - T_nom[i] * I, - Rdot_nom[i][k])).T
                d = - T_nom[i] * Rdot_nom[i][k]
                linearized_ik.UpdateCoefficients(D, d)

        # update trust region
        D = trust_region.GetDenseA()
        lb = T_nom / (1 + kappa)
        ub = T_nom * (1 + kappa)
        trust_region.UpdateCoefficients(D, lb, ub)

        # solve program
        solver = pd.ClarabelSolver()
        result = solver.Solve(prog)
        if not result.is_success():
            return None

        # get solution
        return result.GetSolution(T)
    
    return tangent_solver