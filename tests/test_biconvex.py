import unittest
import numpy as np
from pydrake.all import Hyperrectangle
from scstrajopt.polygonal import polygonal
from scstrajopt.biconvex import biconvex, FixedPositionProgram, FixedVelocityProgram

class TestBiconvex(unittest.TestCase):

    def setUp(self):

        # simple 2d problem
        q_init_1 = np.array([0, 0])
        q_term_1 = np.array([5, 1])
        regions_1 = [
            Hyperrectangle([-1, -2], [2, 3]),
            Hyperrectangle([1, 2], [4, 5]),
            Hyperrectangle([3, 0], [6, 3]),
        ]

        # 2d problem with overlapping knot points
        q_init_2 = np.array([3, 1])
        q_term_2 = np.array([6, 4])
        regions_2 = [
            Hyperrectangle([1, 0], [4, 5]),
            Hyperrectangle([0, 4], [3, 7]),
            Hyperrectangle([2, 3], [7, 6]),
        ]

        # simple 3d problem
        q_init_3 = np.array([1, 1, 0])
        q_term_3 = np.array([1, 1, 5])
        regions_3 = [
            Hyperrectangle([0, 0, 0], [1, 5, 1]),
            Hyperrectangle([0, 4, 0], [5, 5, 1]),
            Hyperrectangle([4, 4, 0], [5, 5, 5]),
            Hyperrectangle([0, 4, 4], [5, 5, 5]),
            Hyperrectangle([0, 0, 4], [1, 5, 5]),
        ]

        # collect problems
        self.q_init = [q_init_1, q_init_2, q_init_3]
        self.q_term = [q_term_1, q_term_2, q_term_3]
        self.regions = [regions_1, regions_2, regions_3]

    def test_biconvex(self):
        decimal = 6
        tol = 10 ** (- decimal)

        # problem parameters
        unit_box = lambda dim: Hyperrectangle([-1] * dim, [1] * dim)
        deg = 5
        time_tol = .1

        # check all problems
        for i in range(3):
            q_init = self.q_init[i]
            q_term = self.q_term[i]
            regions = self.regions[i]
            vel_set = unit_box(len(q_init))
            acc_set = vel_set

            # get polygonal curve
            composite_curve = biconvex(q_init, q_term, regions, vel_set, acc_set, deg, time_tol)
            
            # initial and final conditions
            np.testing.assert_array_almost_equal(q_init, composite_curve.initial_point(), decimal=decimal)
            np.testing.assert_array_almost_equal(q_term, composite_curve.final_point(), decimal=decimal)

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

            # curve duration
            for curve in composite_curve:
                self.assertGreater(curve.duration, time_tol - tol)

            # test decreasing cost
            composite_curve = polygonal(q_init, q_term, regions, vel_set, acc_set, deg, time_tol)
            fixed_position = FixedPositionProgram(regions, vel_set, acc_set, deg, time_tol)
            fixed_velocity = FixedVelocityProgram(q_init, q_term, regions, vel_set, acc_set, deg, time_tol)
            durations = [composite_curve.duration]
            for i in range(3):
                composite_curve = fixed_position.solve(composite_curve)
                durations.append(composite_curve.duration)
                composite_curve = fixed_velocity.solve(composite_curve)
                durations.append(composite_curve.duration)
            durations = np.array(durations)
            self.assertGreater(min(durations[:-1] - durations[1:]), - tol)
            
if __name__ == '__main__':
    unittest.main()