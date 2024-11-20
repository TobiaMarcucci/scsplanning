import numpy as np
from typing import List
from pydrake.all import ConvexSet, MathematicalProgram, eq, ge, ClarabelSolver
from pybezier import BezierCurve, CompositeBezierCurve
from scstrajopt import polygonal

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
    fixed_position = FixedPositionProgram(regions, vel_set, acc_set, deg)
    fixed_velocity = FixedVelocityProgram(q_init, q_term, regions, vel_set, acc_set, deg)

    # alternate until curve duration does not decrease sufficiently
    rel_improvement = lambda duration, curve: (duration - curve.duration) / curve.duration
    position_duration = np.inf
    velocity_duration = curve.duration

    while True:

        # fixed transition points 
        curve = fixed_position.solve(curve)
        if rel_improvement(position_duration, curve) < tol:
            break
        position_duration = curve.duration

        # fixed transition velocities 
        curve = fixed_velocity.solve(curve)
        if rel_improvement(velocity_duration, curve) < tol:
            break
        velocity_duration = curve.duration

    return curve

class BaseProgram(MathematicalProgram):
    '''
    Class that serves as a skeleton for the two programs solved in the biconvex
    alternations.
    '''

    def __init__(self, regions, acc_set, deg):
        super().__init__()

        # default solver
        self.solver = ClarabelSolver()

        # decision variables
        reg = len(regions)
        dim = acc_set.ambient_dimension()
        self.Q = self.add_variables(reg, deg + 1, dim) # position
        self.V = self.add_variables(reg, deg, dim) # velocity
        self.A = self.add_variables(reg, deg - 1, dim) # acceleration
        self.T = self.add_variables(reg) # durations or inverse durations
        self.S = self.add_variables(reg) # acceleration scaling

        # finite difference for velocity
        I = np.eye(dim)
        D = np.hstack((I, -I*deg, I*deg))
        d = np.zeros(dim)
        for i in range(reg):
            for k in range(deg):
                vars = np.concatenate((self.V[i, k], self.Q[i,k+1], self.Q[i, k]))
                self.AddLinearEqualityConstraint(D, d, vars)

        # finite difference for acceleration
        D = np.hstack((I, -I*(deg-1), I*(deg-1)))
        for i in range(reg):
            for k in range(deg-1):
                vars = np.concatenate((self.A[i, k], self.V[i, k+1], self.V[i, k]))
                self.AddLinearEqualityConstraint(D, d, vars)

        # scaling constraint on the acceleration
        for i in range(reg):
            for a in self.A[i]:
                acc_set.AddPointInNonnegativeScalingConstraints(self, a, self.S[i])

        # parametric constraints that decide the value of the acceleration scaling
        D = [[0, 0]]
        d = [0]
        self.scaling_constr = []
        for vars in zip(self.S, self.T):
            c = self.AddLinearEqualityConstraint(D, d, vars)
            self.scaling_constr.append(c.evaluator())

    def add_variables(self, *args):
        '''
        Allows to add multidimensional (tensor) variables to a program.
        '''

        x = np.full(args, None)
        self._fill_array_with_variables(x)

        return x

    def _fill_array_with_variables(self, x):

        if len(x.shape) <= 2:
            x[:] = self.NewContinuousVariables(*x.shape)
        else:
            for xi in x:
                self._fill_array_with_variables(xi)

    def get_solution(self, x, result):
        '''
        Allows to get the optimal value of a multidimensional variable.
        '''

        x_opt = x.copy()
        self._fill_array_with_values(x_opt, result)

        return x_opt

    def _fill_array_with_values(self, x, result):

        if len(x.shape) <= 2:
            x[:] = result.GetSolution(x)
        else:
            for xi in x:
                self._fill_array_with_values(xi, result)

    def parametric_knot_constraints(self, X):

        reg = X.shape[0]
        dim = X.shape[-1]
        D = np.zeros((dim, dim + 1))
        d = np.zeros(dim)
        init_constr = []
        term_constr = []
        for i in range(reg):
            vars = np.concatenate((X[i, 0], [self.T[i]]))
            c = self.AddLinearEqualityConstraint(D, d, vars)
            init_constr.append(c.evaluator())
            vars = np.concatenate((X[i, -1], [self.T[i]]))
            c = self.AddLinearEqualityConstraint(D, d, vars)
            term_constr.append(c.evaluator())

        return init_constr, term_constr

    def update_knot_constraints(self, knots):

        reg = knots.shape[0] - 1
        dim = knots.shape[1]
        D = np.vstack((np.eye(dim), - knots[0])).T
        d = np.zeros(dim)
        for i in range(reg):
            self.init_constr[i].UpdateCoefficients(D, d)
            D[:, -1] = - knots[i + 1]
            self.term_constr[i].UpdateCoefficients(D, d)
   
    def reconstruct_curve(self, Q_opt, T_opt):

        reg = len(T_opt)
        curves = []
        t_start = 0
        for i in range(reg):
            t_stop = t_start + T_opt[i]
            curves.append(BezierCurve(Q_opt[i], t_start, t_stop))
            t_start = t_stop

        return CompositeBezierCurve(curves)

class FixedPositionProgram(BaseProgram):

    def __init__(self, regions, vel_set, acc_set, deg):
        super().__init__(regions, acc_set, deg)

        # minimize total time (in this program T contains the inverse of the
        # durations of the Bezier curves)
        reg = len(regions)
        S = self.NewContinuousVariables(reg)
        self.AddLinearCost([1] * reg, 0, S)
        D = [[1, 0], [0, 1], [0, 0]]
        d = [0, 0, 1]
        for vars in zip(self.T, S):
            self.AddRotatedLorentzConeConstraint(D, d, vars)

        # initial and final velocities are zero
        dim = vel_set.ambient_dimension()
        I = np.eye(dim)
        d = [0] * dim
        self.AddLinearEqualityConstraint(I, d, self.V[0, 0])
        self.AddLinearEqualityConstraint(I, d, self.V[-1, -1])

        # velocity continuity
        D = np.hstack((I, -I))
        for Vi, Vj in zip(self.V[:-1], self.V[1:]):
            vars = np.concatenate((Vi[-1], Vj[0]))
            self.AddLinearEqualityConstraint(D, d, vars)

        # control points must be in corresponding region (avoids overconstraining)
        for i in range(reg):
            if i == 0:
                Qi = self.Q[i, 2:-1] # skip first, second, and last
            elif i == reg - 1:
                Qi = self.Q[i, 1:-2] # skip first, second to last, and last
            else:
                Qi = self.Q[i, 1:-1] # skip first and last
            for q in Qi:
                regions[i].AddPointInNonnegativeScalingConstraints(self, q, self.T[i])

        # velocity constraints (avoids overconstraining)
        for i in range(reg):
            if i == 0:
                Vi = self.V[i, 1:-1]
            else:
                Vi = self.V[i, :-1]
            for v in Vi:
                vel_set.AddPointInSetConstraints(self, v)

        # parametric constraints on knot positions
        self.init_constr, self.term_constr = self.parametric_knot_constraints(self.Q)
                
    def solve(self, curve):

        # extract parameters from given curve
        T_nom = curve.durations()
        knots = curve.knot_points()
        
        # update knot position constraints
        self.update_knot_constraints(knots)

        # update accceleration constraint
        for i, Ti in enumerate(T_nom):
            D = np.array([[1, Ti ** 2]])
            d = np.array([2 * Ti])
            self.scaling_constr[i].UpdateCoefficients(D, d)

        # solve program
        result = self.solver.Solve(self)

        # reconstruct curve
        assert result.is_success()
        T_opt = 1 / result.GetSolution(self.T)
        Q_opt = T_opt[:, None, None] * self.get_solution(self.Q, result)

        return self.reconstruct_curve(Q_opt, T_opt)

class FixedVelocityProgram(BaseProgram):

    def __init__(self, q_init, q_term, regions, vel_set, acc_set, deg):
        super().__init__(regions, acc_set, deg)

        # minimize total time
        reg = len(regions)
        self.AddLinearCost([1] * reg, 0, self.T)

        # initial and final positions
        dim = vel_set.ambient_dimension()
        I = np.eye(dim)
        self.AddLinearEqualityConstraint(I, q_init, self.Q[0, 0])
        self.AddLinearEqualityConstraint(I, q_term, self.Q[-1, -1])

        # position continuity
        D = np.hstack((I, -I))
        d = [0] * dim
        for Qi, Qj in zip(self.Q[:-1], self.Q[1:]):
            vars = np.concatenate((Qi[-1], Qj[0]))
            self.AddLinearEqualityConstraint(D, d, vars)

        # curve control points must be in corresponding region
        for i, region in enumerate(regions):
            if i == 0:
                Qi = self.Q[i, 1:]
            if i == reg - 1:
                Qi = self.Q[i, :-1]
            else:
                Qi = self.Q[i]
            for q in Qi:
                region.AddPointInSetConstraints(self, q)

        # velocity constraints
        for i, Vi in enumerate(self.V):
            # do not constraint first and last point to avoid numerical issues
            for v in Vi[1:-1]:
                vel_set.AddPointInNonnegativeScalingConstraints(self, v, self.T[i])

        # parametric constraints on knot velocities
        self.init_constr, self.term_constr = self.parametric_knot_constraints(self.V)

    def solve(self, curve):

        # extract parameters from given curve
        T_nom = curve.durations()
        knots = curve.derivative().knot_points()

        # update knot velocity constraints
        self.update_knot_constraints(knots)

        # update accceleration constraint
        for i, Ti in enumerate(T_nom):
            D = np.array([[1, - 2 * Ti]])
            d = np.array([- Ti ** 2])
            self.scaling_constr[i].UpdateCoefficients(D, d)

        # solve program
        result = self.solver.Solve(self)

        # reconstruct curve
        assert result.is_success()
        T_opt = result.GetSolution(self.T)
        Q_opt = self.get_solution(self.Q, result)

        return self.reconstruct_curve(Q_opt, T_opt)