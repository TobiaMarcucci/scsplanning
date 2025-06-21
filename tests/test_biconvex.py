import unittest
import numpy as np
from pydrake.all import Hyperrectangle
from scsplanning.polygonal import polygonal
from scsplanning.biconvex import biconvex, FixedPoints, FixedVelocities

class TestBiconvex(unittest.TestCase):

    def setUp(self):

        # simple 2d problem
        q_init_2d = np.array([0, 0])
        q_term_2d = np.array([5, 1])
        regions_2d = [
            Hyperrectangle([-1, -2], [2, 3]),
            Hyperrectangle([1, 2], [4, 5]),
            Hyperrectangle([3, 0], [6, 3]),
        ]
        problem_2d = (q_init_2d, q_term_2d, regions_2d)

        # simple 3d problem
        q_init_3d = np.array([1, 1, 0])
        q_term_3d = np.array([1, 1, 5])
        regions_3d = [
            Hyperrectangle([0, 0, 0], [1, 5, 1]),
            Hyperrectangle([0, 4, 0], [5, 5, 1]),
            Hyperrectangle([4, 4, 0], [5, 5, 5]),
            Hyperrectangle([0, 4, 4], [5, 5, 5]),
            Hyperrectangle([0, 0, 4], [1, 5, 5]),
        ]
        problem_3d = (q_init_3d, q_term_3d, regions_3d)

        # collect problems
        self.problems = [problem_2d, problem_3d]

    def test_biconvex(self):
        decimal = 6
        tol = 10 ** (- decimal)

        # problem parameters
        unit_box = lambda dim: Hyperrectangle([-1] * dim, [1] * dim)
        deg = 5

        # check all problems
        for problem in self.problems:
            q_init, q_term, regions = problem
            vel_set = unit_box(len(q_init))
            acc_set = vel_set

            # get polygonal curve
            composite_curve = biconvex(q_init, q_term, regions, vel_set, acc_set, deg)
            
            # initial and final conditions
            np.testing.assert_array_almost_equal(q_init, composite_curve.initial_point, decimal=decimal)
            np.testing.assert_array_almost_equal(q_term, composite_curve.final_point, decimal=decimal)

            # control points are in convex regions
            for k, curve in enumerate(composite_curve):
                for q in curve.points:
                    self.assertTrue(regions[k].PointInSet(q, tol))

            # velocity constraints
            velocity = composite_curve.derivative()
            for k, curve in enumerate(velocity):
                for p in curve.points:
                    self.assertTrue(vel_set.PointInSet(p, tol))

            # acceleration constraints
            acceleration = composite_curve.derivative()
            for k, curve in enumerate(acceleration):
                for p in curve.points:
                    self.assertTrue(acc_set.PointInSet(p, tol))

            # test decreasing cost
            n_iters = 4
            composite_curve = polygonal(q_init, q_term, regions, vel_set, acc_set, deg)
            fixed_position = FixedPoints(regions, vel_set, acc_set, deg)
            fixed_velocity = FixedVelocities(q_init, q_term, regions, vel_set, acc_set, deg)
            durations = [composite_curve.duration]
            for i in range(n_iters):
                composite_curve = fixed_position.solve(composite_curve)
                durations.append(composite_curve.duration)
                composite_curve = fixed_velocity.solve(composite_curve)
                durations.append(composite_curve.duration)
            durations = np.array(durations)
            self.assertGreater(min(durations[:-1] - durations[1:]), - tol)
            
if __name__ == '__main__':
    unittest.main()
