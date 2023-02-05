from z3 import *
import numpy as np

(x_0, y_0) = (2, 0)
(x_c, y_c) = (4, 6)
r = 2
visual_depth = 8

s = Solver()
x, y = Real('x'), Real('y')

y_rng = y - y_0
x_rng = x - x_0

# Find intersection points given these equations and constraints:
# - Equally spaced concurrent lines with center (x_0, y_0):
#       (x - x_0)sin(a) = (y - y_0)cos(a) for a in [0, pi]
# - Constraint x and y to the maximum visual depth:
#       |y - y_0| < visual_depth, |x - x_0| < visual_depth
# - Circumference of center (x_c, y_c) and radius r:
#       (x - x_c)^2 + (y - y_c)^2 = r^2
s.add(And(Or([(x - x_0) * np.sin(a) == (y - y_0) * np.cos(a) for a in np.linspace(0, np.pi, 7)]),
          (x - x_c) ** 2 + (y - y_c) ** 2 == r ** 2,
          If(y_rng >= 0, y_rng, -y_rng) <= visual_depth,
          If(x_rng >= 0, x_rng, -x_rng) <= visual_depth))

# If it's satisfiable the agent 'c' is on the view of agent '0'
if s.check() == sat:
    # Compute the distance between the two agents e.g. between their center points
    d = np.sqrt((x_0 - x_c) ** 2 + (y_0 - y_c) ** 2)
    print(d)
