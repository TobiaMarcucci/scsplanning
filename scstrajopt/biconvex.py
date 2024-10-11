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
    time_tol: float = 1e-2,
    debug_mode: bool = False
    ) -> CompositeBezierCurve:

    # in debug mode we check all the assumptions on the problem data
    if debug_mode:
        check_input_data(q_init, q_term, regions, vel_set, acc_set, deg, time_tol)

    # compute initial guess
    curve = polygonal(q_init, q_term, regions, vel_set, acc_set, deg, time_tol)

    # instantiate programs for the biconvex alternation
    fixed_position = FixedPositionProgram(regions, vel_set, acc_set, deg, time_tol)
    fixed_velocity = FixedVelocityProgram(q_init, q_term, regions, vel_set, acc_set, deg, time_tol)

    # alternate until curve duration does not decrease sufficiently
    while True:
        prev_duration = curve.duration
        curve = fixed_position.solve(curve)
        curve = fixed_velocity.solve(curve)
        rel_improvement = (prev_duration - curve.duration) / curve.duration
        if rel_improvement < time_tol:
            break

    return curve

def check_input_data(q_init, q_term, regions, vel_set, acc_set, deg, time_tol):

    # vectors and sets must all have the same dimensions
    assert len(q_init.shape) == 1
    assert len(q_term.shape) == 1
    dim = len(q_init)
    assert len(q_term) == dim
    assert all(region.ambient_dimension() == dim for region in regions)
    assert vel_set.ambient_dimension() == dim
    assert acc_set.ambient_dimension() == dim

    # tolerance must be positive
    assert time_tol > 0

    # curve degree must be high enough to represent straight lines with zero
    # velocity at the endpoints, so that algorithm is guaranteed to succeed
    assert deg >= 3

    # initial and final points must be in the first and last region respectively
    assert regions[0].PointInSet(q_init)
    assert regions[-1].PointInSet(q_term)

    # derivative constraint set must contain the origin in teir interior
    simplex_vertices = np.vstack((np.zeros(dim), np.eye(dim)))
    simplex_vertices -= np.mean(simplex_vertices, axis=0)
    simplex_vertices *= 1e-4
    for p in simplex_vertices:
        assert vel_set.PointInSet(p)
        assert acc_set.PointInSet(p)

    # consecutive sets must intersect
    for region1, region2 in zip(regions[1:], regions[:-1]):
        assert region1.IntersectsWith(region2)

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

        # finite difference for computing the control points of the derivatives
        diff = lambda X: (X[1:] - X[:-1]) * (len(X) - 1)
        for i in range(reg):
            self.AddLinearConstraint(eq(self.V[i], diff(self.Q[i])))
            self.AddLinearConstraint(eq(self.A[i], diff(self.V[i])))

        # scaling constraint on the acceleration
        for i in range(reg):
            for a in self.A[i]:
                acc_set.AddPointInNonnegativeScalingConstraints(self, a, self.S[i])

        # parametric constraints that decide the value of the acceleration scaling
        lhs = np.zeros((1, 2))
        rhs = np.zeros(1)
        self.scaling_constr = []
        for i in range(reg):
            vars = np.array((self.S[i], self.T[i]))
            self.scaling_constr.append(self.AddLinearEqualityConstraint(lhs, rhs, vars).evaluator())

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
        lhs = np.zeros((dim, dim + 1))
        rhs = np.zeros(dim)
        init_constr = []
        term_constr = []
        for i in range(reg):
            vars = np.concatenate((X[i, 0], [self.T[i]]))
            c = self.AddLinearEqualityConstraint(lhs, rhs, vars)
            init_constr.append(c.evaluator())
            vars = np.concatenate((X[i, -1], [self.T[i]]))
            c = self.AddLinearEqualityConstraint(lhs, rhs, vars)
            term_constr.append(c.evaluator())

        return init_constr, term_constr

    def update_knot_constraints(self, knots):

        reg = knots.shape[0] - 1
        dim = knots.shape[1]
        lhs = np.vstack((np.eye(dim), - knots[0])).T
        rhs = np.zeros(dim)
        for i in range(reg):
            self.init_constr[i].UpdateCoefficients(lhs, rhs)
            lhs[:, -1] = - knots[i + 1]
            self.term_constr[i].UpdateCoefficients(lhs, rhs)
   
    def reconstruct_curve(self, Q_opt, T_opt):

        reg = len(T_opt)
        beziers = []
        t_start = 0
        for i in range(reg):
            t_stop = t_start + T_opt[i]
            beziers.append(BezierCurve(Q_opt[i], t_start, t_stop))
            t_start = t_stop

        return CompositeBezierCurve(beziers)


class FixedPositionProgram(BaseProgram):

    def __init__(self, regions, vel_set, acc_set, deg, time_tol):
        super().__init__(regions, acc_set, deg)

        # minimize total time (in this program T contains the inverse of the
        # durations of the Bezier curves)
        S = self.NewContinuousVariables(len(regions))
        self.AddLinearCost(sum(S))
        for Ti, Si in zip(self.T, S):
            self.AddRotatedLorentzConeConstraint(Si, Ti, 1)
            self.AddLinearConstraint(Ti <= 1 / time_tol)

        # initial and final velocities are zero
        self.AddLinearConstraint(eq(self.V[0, 0], 0))
        self.AddLinearConstraint(eq(self.V[-1, -1], 0))

        # velocity is continuous when moving between one region and the next
        for Vi, Vj in zip(self.V[:-1], self.V[1:]):
            self.AddLinearConstraint(eq(Vi[-1], Vj[0]))

        # control points must be in corresponding region (avoids overconstraining)
        for i in range(len(regions)):
            if i == 0:
                Qi = self.Q[i, 2:-1] # skip first, second, and last
            elif i == len(regions) - 1:
                Qi = self.Q[i, 1:-2] # skip first, second to last, and last
            else:
                Qi = self.Q[i, 1:-1] # skip first and last
            for q in Qi:
                regions[i].AddPointInNonnegativeScalingConstraints(self, q, self.T[i])

        # velocity constraints (avoids overconstraining)
        for i in range(len(regions)):
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
            lhs = np.array([[1, Ti ** 2]])
            rhs = np.array([2 * Ti])
            self.scaling_constr[i].UpdateCoefficients(lhs, rhs)

        # solve program
        result = self.solver.Solve(self)

        # reconstruct curve
        assert result.is_success()
        T_opt = 1 / result.GetSolution(self.T)
        Q_opt = T_opt[:, None, None] * self.get_solution(self.Q, result)

        return self.reconstruct_curve(Q_opt, T_opt)


class FixedVelocityProgram(BaseProgram):

    def __init__(self, q_init, q_term, regions, vel_set, acc_set, deg, time_tol):
        super().__init__(regions, acc_set, deg)

        # time durations are nonnegative
        self.AddLinearConstraint(ge(self.T, time_tol))

        # minimize total time
        self.AddLinearCost(sum(self.T))

        # initial and final positions
        self.AddLinearConstraint(eq(self.Q[0, 0], q_init))
        self.AddLinearConstraint(eq(self.Q[-1, -1], q_term))

        # position is continuous when moving between one region and the next
        for Qi, Qj in zip(self.Q[:-1], self.Q[1:]):
            self.AddLinearConstraint(eq(Qi[-1], Qj[0]))

        # curve control points must be in corresponding region
        for i, Qi in enumerate(self.Q):
            for q in Qi:
                regions[i].AddPointInSetConstraints(self, q)

        # velocity constraints
        if vel_set is not None:
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
            lhs = np.array([[1, - 2 * Ti]])
            rhs = np.array([- Ti ** 2])
            self.scaling_constr[i].UpdateCoefficients(lhs, rhs)

        # solve program
        result = self.solver.Solve(self)

        # reconstruct curve
        assert result.is_success()
        T_opt = result.GetSolution(self.T)
        Q_opt = self.get_solution(self.Q, result)

        return self.reconstruct_curve(Q_opt, T_opt)
