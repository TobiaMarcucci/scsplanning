import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import Hyperrectangle
from matplotlib.patches import Rectangle
from scstrajopt import biconvex

# initial and terminal positions
q_init = np.array([1, -1])
q_term = np.array([4, 1])

# convex regions to be traversed
regions = [
    Hyperrectangle([0, -2], [2, 3]),
    Hyperrectangle([1, 2], [4, 4]),
    Hyperrectangle([3, 0], [5, 3]),
]

# velocity and acceleration constraints
vel_set = Hyperrectangle([-1, -1], [1, 1])
acc_set = Hyperrectangle([-1, -1], [1, 1])

# degree of the Bezier curves
deg = 5

# optimize curve
curve = biconvex(q_init, q_term, regions, vel_set, acc_set, deg)

# plot curve
plt.figure()
for region in regions:
    diagonal = region.ub() - region.lb()
    rect = Rectangle(region.lb(), *diagonal, fc='w', ec='k')
    plt.gca().add_patch(rect)
curve.plot_trace_2d()
plt.show()