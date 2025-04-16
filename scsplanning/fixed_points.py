import numpy as np
from scsplanning.base_program import BaseProgram

class FixedPoints(BaseProgram):

    def __init__(self, regions, vel_set, acc_set, deg):
        super().__init__(regions, deg)

        # problem data
        reg = len(regions)
        dim = regions[0].ambient_dimension()

        # additional variables
        self.T = self.add_variables(reg) # traversal times
        self.S = self.add_variables(reg) # inverse of traversal times
        self.acc_sc = self.add_variables(reg) # acceleration scaling
        
        # minimize total time
        self.AddLinearCost([1] * reg, 0, self.T)
        D = [[1, 0], [0, 1], [0, 0]]
        d = [0, 0, 1]
        for vars in zip(self.S, self.T):
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
        for si, Ai in zip(self.acc_sc, self.A):
            for a in Ai:
                acc_set.AddPointInNonnegativeScalingConstraints(self, a, si)

        # parametric constraints for the value of the acceleration scaling
        D = [[0, 0]]
        d = [0]
        self.scaling_constr = []
        for vars in zip(self.acc_sc, self.S):
            c = self.AddLinearEqualityConstraint(D, d, vars)
            self.scaling_constr.append(c.evaluator())
                
    def solve(self, curve):

        # update transition-point constraints
        transition_points = curve.transition_points()
        self.update_transition_constraints(transition_points)

        # update accceleration constraint
        traversal_times = curve.durations()
        for Ti, ci in zip(traversal_times, self.scaling_constr):
            D = [[1, Ti ** 2]]
            d = [2 * Ti]
            ci.UpdateCoefficients(D, d)

        # solve program
        result = self.solver.Solve(self)
        assert result.is_success()

        # reconstruct curve
        T_opt = result.GetSolution(self.T)
        Q_opt =  T_opt[:, None, None] * self.get_solution(self.Q, result)

        return self.reconstruct_curve(Q_opt, T_opt)
