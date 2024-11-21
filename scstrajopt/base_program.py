import numpy as np
from pydrake.all import MathematicalProgram, ClarabelSolver
from pybezier import BezierCurve, CompositeBezierCurve

class BaseProgram(MathematicalProgram):
    '''
    Class that serves as a skeleton for the two programs solved in the biconvex
    alternations.
    '''

    def __init__(self, regions, deg):
        super().__init__()

        # problem data
        reg = len(regions)
        dim = regions[0].ambient_dimension()
        self.solver = ClarabelSolver()

        # control points
        self.Q = self.add_variables(reg, deg + 1, dim) # position
        self.V = self.add_variables(reg, deg, dim) # velocity
        self.A = self.add_variables(reg, deg - 1, dim) # acceleration

        # finite difference for derivatives
        def finite_difference(X, Y):
            deg, dim = Y[0].shape
            I = np.eye(dim)
            D = np.hstack((I, -I * deg, I * deg))
            d = np.zeros(dim)
            for Xi, Yi in zip(X, Y):
                for k in range(deg):
                    vars = np.concatenate((Yi[k], Xi[k+1], Xi[k]))
                    self.AddLinearEqualityConstraint(D, d, vars)
        finite_difference(self.Q, self.V)
        finite_difference(self.V, self.A)

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

    def parametric_transition_constraints(self, Q, T):

        dim = Q.shape[-1]
        D = np.eye(dim, dim + 1)
        d = np.zeros(dim)
        self.init_constr = []
        self.term_constr = []
        for Qi, Ti in zip(Q, T):

            # first control point
            vars = np.concatenate((Qi[0], [Ti]))
            c = self.AddLinearEqualityConstraint(D, d, vars)
            self.init_constr.append(c.evaluator())

            # last control point
            vars = np.concatenate((Qi[-1], [Ti]))
            c = self.AddLinearEqualityConstraint(D, d, vars)
            self.term_constr.append(c.evaluator())

    def update_transition_constraints(self, points):

        D = self.init_constr[0].GetDenseA()
        d = np.zeros(D.shape[0])
        D[:, -1] = - points[0]
        for i, point in enumerate(points[1:]):
            self.init_constr[i].UpdateCoefficients(D, d)
            D[:, -1] = - point
            self.term_constr[i].UpdateCoefficients(D, d)
   
    def reconstruct_curve(self, Q_opt, T_opt):

        curves = []
        t_start = 0
        for Qi, Ti in zip(Q_opt, T_opt):
            t_stop = t_start + Ti
            curves.append(BezierCurve(Qi, t_start, t_stop))
            t_start = t_stop

        return CompositeBezierCurve(curves)