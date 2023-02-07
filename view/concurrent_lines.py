from datetime import datetime

from z3 import *
import numpy as np

(x_0, y_0) = (0, 0)
(x_c, y_c) = (-3, -2)
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
l_eq = [(x - x_0) * np.sin(a) - (y - y_0) * np.cos(a) == 0 for a in np.linspace(0, np.pi, 7)]
d = []
for i in range(len(l_eq)):
    o = Optimize()
    o.add(And(l_eq[i],
              (x - x_c) ** 2 + (y - y_c) ** 2 - r ** 2 == 0,
              If(y_rng >= 0, y_rng, -y_rng) <= visual_depth,
              If(x_rng >= 0, x_rng, -x_rng) <= visual_depth))

    if y_c > y_0:
        o.minimize(x)
    elif y_c > y_0:
        o.maximize(y)
    else:
        if x_c > x_0:
            o.minimize(x)
        else:
            o.maximize(x)

    if o.check() == sat:
        m = o.model()

        if isinstance(m[x], AlgebraicNumRef):
            mx, my = m[x].approx(10), m[y].approx(10)
        else:
            mx, my = m[x], m[y]

        x_p, y_p = float(mx.numerator_as_long()) / float(mx.denominator_as_long()), \
            float(my.numerator_as_long()) / float(my.denominator_as_long())

        # Compute the distance between the agent center (x_0, y_0)
        # and the intersection point which is closer to it
        d.append(np.sqrt((x_0 - x_p) ** 2 + (y_0 - y_p) ** 2))
    else:
        d.append(-1)

print(d)
