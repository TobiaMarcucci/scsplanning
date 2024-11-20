import numpy as np
from scstrajopt.base_program import BaseProgram

class FixedPoints(BaseProgram):

    def __init__(self, regions, vel_set, acc_set, deg, relaxation=False):
        super().__init__(regions, deg)

        # problem data
        reg = len(regions)
        dim = regions[0].ambient_dimension()
        self.relaxation = relaxation

        # additional variables
        self.S = self.add_variables(reg) # inverse of traversal times
        self.T = self.add_variables(reg) # acceleration scaling

        # minimize total time
        if self.relaxation:
            vars = self.T
        else:
            vars = self.NewContinuousVariables(reg)
        self.AddLinearCost([1] * reg, 0, vars)
        D = [[1, 0], [0, 1], [0, 0]]
        d = [0, 0, 1]
        for vars in zip(self.S, vars):
            self.AddRotatedLorentzConeConstraint(D, d, vars)

        # initial and final velocities are zero
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
        for i, region in enumerate(regions):
            if i == 0:
                Qi = self.Q[i, 2:-1] # skip first, second, and last
            elif i == reg - 1:
                Qi = self.Q[i, 1:-2] # skip first, second to last, and last
            else:
                Qi = self.Q[i, 1:-1] # skip first and last
            for q in Qi:
                region.AddPointInNonnegativeScalingConstraints(self, q, self.S[i])

        # velocity constraints (avoids overconstraining)
        for i in range(reg):
            if i == 0:
                Vi = self.V[i, 1:-1]
            else:
                Vi = self.V[i, :-1]
            for v in Vi:
                vel_set.AddPointInSetConstraints(self, v)

        # parametric constraints on transition points
        self.parametric_transition_constraints(self.Q, self.S)

        # scaling constraint on the acceleration
        for Ai, Ti in zip(self.A, self.T):
            for a in Ai:
                acc_set.AddPointInNonnegativeScalingConstraints(self, a, Ti)

        # parametric constraints for the value of the acceleration scaling
        if not self.relaxation:
            D = [[0, 0]]
            d = [0]
            self.scaling_constr = []
            for vars in zip(self.T, self.S):
                c = self.AddLinearEqualityConstraint(D, d, vars)
                self.scaling_constr.append(c.evaluator())
                
    def solve(self, curve):

        # update knot position constraints
        transition_points = curve.transition_points()
        self.update_transition_constraints(transition_points)

        # update accceleration constraint
        if not self.relaxation:
            traversal_times = curve.durations()
            for Ti, ci in zip(traversal_times, self.scaling_constr):
                D = [[1, Ti ** 2]]
                d = [2 * Ti]
                ci.UpdateCoefficients(D, d)

        # solve program
        result = self.solver.Solve(self)
        assert result.is_success()

        # reconstruct curve
        S_opt = result.GetSolution(self.S)
        Q_opt = self.get_solution(self.Q, result)
        traversal_times = 1 / S_opt
        points =  traversal_times[:, None, None] * Q_opt
        if self.relaxation: # slow down trajectory if infeasible
            T_opt = result.GetSolution(self.T)
            traversal_times *= max(T_opt * S_opt) ** .5

        return self.reconstruct_curve(points, traversal_times)