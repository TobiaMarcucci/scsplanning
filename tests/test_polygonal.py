import unittest
import numpy as np
from pydrake.all import Hyperrectangle
from scstrajopt.polygonal import (polygonal_curve, get_knots, get_kink_indices,
                                  get_crossing_time, get_point_to_point_solver)

class TestPolygonalCurve(unittest.TestCase):

    def test_polygonal_curve(self):
        pass

    def test_get_knots(self):
        decimal = 4

        # simple 2d problem
        q_init = np.array([0, 0])
        q_term = np.array([5, 1])
        regions = [
            Hyperrectangle([-1, -2], [2, 3]),
            Hyperrectangle([1, 2], [5, 5]),
            Hyperrectangle([3, 0], [6, 3]),
        ]
        knots = get_knots(q_init, q_term, regions)
        self.assertEqual(knots.shape, (4, 2))
        target_knots = np.array([q_init, [2, 2], [3, 2], q_term])
        for knot, target_knot in zip(knots, target_knots):
            np.testing.assert_array_almost_equal(knot, target_knot, decimal=decimal)

        # simple 3d problem
        q_init = np.array([1, 1, 0])
        q_term = np.array([1, 1, 5])
        regions = [
            Hyperrectangle([0, 0, 0], [1, 5, 1]),
            Hyperrectangle([0, 4, 0], [5, 5, 1]),
            Hyperrectangle([4, 4, 0], [5, 5, 5]),
            Hyperrectangle([0, 4, 4], [5, 5, 5]),
            Hyperrectangle([0, 0, 4], [1, 5, 5]),
        ]
        knots = get_knots(q_init, q_term, regions)
        self.assertEqual(knots.shape, (6, 3))
        target_knots = np.array([q_init, [1, 4, .5], [4, 4, 1], [4, 4, 4], [1, 4, 4.5], q_term])
        for knot, target_knot in zip(knots, target_knots):
            np.testing.assert_array_almost_equal(knot, target_knot, decimal=decimal)

        # 2d problem with overlapping knot points
        q_init = np.array([3, 1])
        q_term = np.array([6, 4])
        regions = [
            Hyperrectangle([1, 0], [4, 5]),
            Hyperrectangle([0, 4], [3, 7]),
            Hyperrectangle([2, 3], [7, 6]),
        ]
        knots = get_knots(q_init, q_term, regions)
        self.assertEqual(knots.shape, (4, 2))
        target_knots = np.array([q_init, [3, 4], [3, 4], q_term])
        for knot, target_knot in zip(knots, target_knots):
            np.testing.assert_array_almost_equal(knot, target_knot, decimal=decimal)
        
if __name__ == '__main__':
    unittest.main()