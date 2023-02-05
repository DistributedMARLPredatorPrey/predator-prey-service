from z3 import *
import numpy as np

(x_0, y_0) = (2, 0)
(x_c, y_c) = (4, 6)
r = 2
lidar_len = 8

s = Solver()
x, y = Real('x'), Real('y')

y_rng = y - y_0
x_rng = x - x_0

s.add(And(Or([(x - x_0) * np.sin(a) == (y - y_0) * np.cos(a) for a in np.linspace(0, np.pi, 7)]),
          (x - x_c) ** 2 + (y - y_c) ** 2 == r ** 2,
          If(y_rng >= 0, y_rng, -y_rng) <= lidar_len,
          If(x_rng >= 0, x_rng, -x_rng) <= lidar_len))

if s.check() == sat:
    d = np.sqrt((x_0 - x_c) ** 2 + (y_0 - y_c) ** 2)
    print(d)
