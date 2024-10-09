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

        q_init = np.array([0, 0])
        q_term = np.array([5, 1])
        regions = [
            Hyperrectangle([-1, -2], [2, 3]),
            Hyperrectangle([1, 2], [5, 5]),
            Hyperrectangle([3, 0], [6, 3]),
        ]
        knots = get_knots(q_init, q_term, regions)
        self.assertEqual(knots.shape[0], 4)
        self.assertEqual(knots.shape[1], 2)
        np.testing.assert_array_almost_equal(knots[0], q_init, decimal=decimal)
        np.testing.assert_array_almost_equal(knots[1], [2, 2], decimal=decimal)
        np.testing.assert_array_almost_equal(knots[2], [3, 2], decimal=decimal)
        np.testing.assert_array_almost_equal(knots[3], q_term, decimal=decimal)

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
        print(knots)
        self.assertEqual(knots.shape[0], 6)
        self.assertEqual(knots.shape[1], 3)
        np.testing.assert_array_almost_equal(knots[0], q_init, decimal=decimal)
        np.testing.assert_array_almost_equal(knots[1], [1, 4, .5], decimal=decimal)
        np.testing.assert_array_almost_equal(knots[2], [4, 4, 1], decimal=decimal)
        np.testing.assert_array_almost_equal(knots[3], [4, 4, 4], decimal=decimal)
        np.testing.assert_array_almost_equal(knots[4], [1, 4, 4.5], decimal=decimal)
        np.testing.assert_array_almost_equal(knots[5], q_term, decimal=decimal)
        
if __name__ == '__main__':
    unittest.main()