from datetime import datetime

from z3 import *
import numpy as np

np.random.seed(42)
(x_0, y_0) = (0, -0.1)
cds = [(xc, yc) for (xc, yc) in
       [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(20)]]

r = 0.1
vd = 0.4

x, y = Real('x'), Real('y')

y_rng = y - y_0
x_rng = x - x_0

# Find intersection points given these equations and constraints:
# - Equally spaced concurrent lines with center (x_0, y_0):
#       (x - x_0)sin(a) = (y - y_0)cos(a) for a in [0, pi]
# - Constraint x and y to the maximum visual depth:
#       |y - y_0| < vd, |x - x_0| < vd
# - Circumference of center (x_c, y_c) and radius r:
#       (x - x_c)^2 + (y - y_c)^2 = r^2

t_start = datetime.now()
y_ineq = [y >= 0, y < 0]
x_ineq = [x >= 0, x < 0]
d = []

m = []
for mm in [1/2, x, 2]:
    m.append(mm)
    m.append(-mm)

for i in range(len(y_ineq)):
    # for a in np.linspace(0, np.pi, 7):
    for mm in m:
        o = Optimize()
        """
        o.add(And((x - x_0) * np.sin(a) - (y - y_0) * np.cos(a) == 0,
                  Or([And(y - cy - (np.cos(a) / np.sin(a) * (x - cx)) == 0 if a != 0 else x == cx,
                          If(x - cx >= 0, x - cx <= r, - (x - cx) <= r),
                          If(y - cy >= 0, y - cy <= r, - (y - cy) <= r))
                      for (cx, cy) in cds]),
                  y_ineq[i],
                  If(y_rng >= 0, y_rng, -y_rng) - vd <= 0,
                  If(x_rng >= 0, x_rng, -x_rng) - vd <= 0))
        """

        o.add(And((y - y_0) - mm * (x - x_0) == 0,
                  # If(mm != 0, y - cy - (1 / mm * (x - cx)) == 0, x == cx)
                  Or([And(y - cy - (1 / mm * (x - cx)) == 0,
                          If(x - cx >= 0, x - cx <= r, - (x - cx) <= r),
                          If(y - cy >= 0, y - cy <= r, - (y - cy) <= r))
                      for (cx, cy) in cds]),
                  y_ineq[i],
                  If(y_rng >= 0, y_rng, -y_rng) - vd <= 0,
                  If(x_rng >= 0, x_rng, -x_rng) - vd <= 0))

        o.minimize(If(y_ineq[0], y, If(And(y_ineq), If(x_ineq[0], x, -x), -y)))

        if o.check() == sat:
            m = o.model()

            mx, my = m[x], m[y]
            if isinstance(mx, AlgebraicNumRef):
                mx = mx.approx(10)
            if isinstance(m[y], AlgebraicNumRef):
                my = my.approx(10)

            x_p, y_p = float(mx.numerator_as_long()) / float(mx.denominator_as_long()), \
                float(my.numerator_as_long()) / float(my.denominator_as_long())

            # Compute the distance between the agent center (x_0, y_0)
            # and the intersection point which is closer to it
            d.append(np.sqrt((x_0 - x_p) ** 2 + (y_0 - y_p) ** 2))
        else:
            d.append(-1)

print(datetime.now() - t_start)
print(d)
