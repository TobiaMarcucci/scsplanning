import numpy as np
from scsplanning.base_program import BaseProgram

class FixedVelocities(BaseProgram):

    def __init__(self, q_init, q_term, regions, vel_set, acc_set, deg):
        super().__init__(regions, deg)

        # problem data
        reg = len(regions)
        dim = regions[0].ambient_dimension()

        # additional variables
        self.T = self.add_variables(reg) # traversal times
        self.acc_sc = self.add_variables(reg) # acceleration scaling

        # minimize trajectory duration
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
                Qi = self.Q[i, 2:] # skip first, second, and last
            elif i == reg - 1:
                Qi = self.Q[i, :-2] # skip second to last and last
            else:
                Qi = self.Q[i] # skip last
            for q in Qi:
                region.AddPointInSetConstraints(self, q)

        # velocity constraints
        for i, Vi in enumerate(self.V):
            # do not constraint first and last point to avoid numerical issues
            for v in Vi[1:-1]:
                vel_set.AddPointInNonnegativeScalingConstraints(self, v, self.T[i])

        # parametric constraints on transition velocities
        self.parametric_transition_constraints(self.V, self.T)

        # scaling constraint on the acceleration
        for si, Ai in zip(self.acc_sc, self.A):
            for a in Ai:
                acc_set.AddPointInNonnegativeScalingConstraints(self, a, si)

        # parametric constraints for the value of the acceleration scaling
        D = [[0, 0]]
        d = [0]
        self.scaling_constr = []
        for vars in zip(self.acc_sc, self.T):
            c = self.AddLinearEqualityConstraint(D, d, vars)
            self.scaling_constr.append(c.evaluator())

    def solve(self, curve):

        # update transition velocity constraints
        transition_velocities = curve.derivative().transition_points()
        self.update_transition_constraints(transition_velocities)

        # update accceleration constraint
        traversal_times = curve.durations()
        for Ti, ci in zip(traversal_times, self.scaling_constr):
            D = [[1, - 2 * Ti]]
            d = [- Ti ** 2]
            ci.UpdateCoefficients(D, d)

        # solve program
        result = self.solver.Solve(self)
        assert result.is_success()

        # reconstruct curve
        T_opt = result.GetSolution(self.T)
        Q_opt = self.get_solution(self.Q, result)

        return self.reconstruct_curve(Q_opt, T_opt)
