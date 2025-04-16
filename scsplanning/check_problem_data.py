import numpy as np
from typing import List
from pydrake.all import ConvexSet

def check_problem_data(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[ConvexSet],
    vel_set: ConvexSet,
    acc_set: ConvexSet,
    deg: int,
    interior_tol: float = 1e-4
    ):

    # vectors and sets must all have the same dimensions
    assert len(q_init.shape) == 1
    assert len(q_term.shape) == 1
    dim = len(q_init)
    assert len(q_term) == dim
    assert all(region.ambient_dimension() == dim for region in regions)
    assert vel_set.ambient_dimension() == dim
    assert acc_set.ambient_dimension() == dim

    # curve degree must be high enough to represent straight lines with zero
    # velocity at the endpoints, so that algorithm is guaranteed to succeed
    assert deg >= 3

    # initial and final points must be in the first and last region respectively
    assert regions[0].PointInSet(q_init)
    assert regions[-1].PointInSet(q_term)

    # derivative constraint set must contain the origin in teir interior
    simplex_vertices = np.vstack((np.zeros(dim), np.eye(dim)))
    simplex_vertices -= np.mean(simplex_vertices, axis=0)
    simplex_vertices *= interior_tol
    for p in simplex_vertices:
        assert vel_set.PointInSet(p)
        assert acc_set.PointInSet(p)

    # consecutive sets must intersect
    for region1, region2 in zip(regions[1:], regions[:-1]):
        assert region1.IntersectsWith(region2)

    # gap between regions separated by two
    for region1, region2 in zip(regions[2:], regions[:-2]):
        assert not region1.IntersectsWith(region2)