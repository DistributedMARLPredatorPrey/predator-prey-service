from datetime import datetime

from z3 import *
import numpy as np

np.random.seed(42)

(x_0, y_0) = (1, 2)
# cds = [(xc, yc) for (xc, yc) in
#       [(np.random.uniform(-5, 5), np.random.uniform(-5, 5)) for _ in range(100)]]
cds = [(3, 1), (5, 4), (-1, 4), (2.5, 2)]
cds.append((0, 0))

r = 0.3
vd = 3

x, y = Real('x'), Real('y')

y_rng = y - y_0
x_rng = x - x_0

# Find intersection points given these equations and constraints:
# - Equally spaced concurrent lines with center (x_0, y_0):
#       (x - x_0) * sin(a) = (y - y_0) * cos(a) for a in [0, pi]
# - Constraint x and y to the maximum visual depth:
#       |y - y_0| < vd, |x - x_0| < vd
# - Box of center (x_c, y_c) and radius r:
#       x_c - r <= x <= x_c + r, y_c - r <= y <= y_c + r

t_start = datetime.now()
# y_ineq = [y >= y_0, y < y_0]
# x_ineq = [x >= x_0, x < x_0]
d = []

rng_cstr = [If(y_rng >= 0, y_rng, - y_rng) - vd < 0,
            If(x_rng >= 0, x_rng, - x_rng) - vd < 0]
t0_start = datetime.now()

for lconstr in [y < y_0, y >= y_0]:

    for a in np.linspace(0, np.pi, 7):
        o = Optimize()
        # (1 / np.sin(a)) * x - np.sin(a) * cx - (y - cy) * np.cos(a) == 0 if a != 0 else x == cx

        # y - cy - (np.cos(a) / np.sin(a) * (x - cx)) == 0 if a != 0 else x == cx,
        # If(x - cx >= 0, x - cx <= r, - (x - cx) <= r),
        # If(y - cy >= 0, y - cy <= r, - (y - cy) <= r))
        o.add(
            And(
                # pencil of lines (set of lines passing through a common point):
                #   (x - x_0) * sin(a) = (y - y_0) * cos(a) forall a in [0, pi]
                (x - x_0) * np.sin(a) - (y - y_0) * np.cos(a) == 0,
                # visual depth
                And(rng_cstr),
                # agent' boxes
                Or([
                    Or(
                        And(x <= cx + r, x >= cx - r, y == cy - r),
                        And(x <= cx + r, x >= cx - r, y == cy + r),
                        And(y <= cy + r, y >= cy - r, x == cx - r),
                        And(y <= cy + r, y >= cy - r, x == cx + r)
                    )
                    for (cx, cy) in cds
                ]),
                # half line constraint
                lconstr
            )
        )

        o.minimize(
            If(y > y_0,
               y,
               If(y < y_0,
                  - y,
                  If(x >= x_0, x, -x)
                  )
               )
        )

        t_start = datetime.now()

        if o.check() == sat:
            model = o.model()

            mx, my = model[x], model[y]
            if isinstance(mx, AlgebraicNumRef):
                mx = mx.approx(10)
            if isinstance(my, AlgebraicNumRef):
                my = my.approx(10)

            x_p, y_p = float(mx.numerator_as_long()) / float(mx.denominator_as_long()), \
                       float(my.numerator_as_long()) / float(my.denominator_as_long())
            print((x_p, y_p))
            # Compute the distance between the agent center (x_0, y_0)
            # and the intersection point which is closer to it
            d.append(np.sqrt(np.power(x_0 - x_p, 2) + np.power(y_0 - y_p, 2)))
            # print(datetime.now() - t_start)
        else:
            # print(datetime.now() - t_start)

            d.append(-1)

print(datetime.now() - t0_start)
print(d)
