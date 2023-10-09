from datetime import datetime
from typing import List, Tuple

from z3 import *
import numpy as np
from model.agent import Agent
from model.environment import Environment

np.random.seed(42)


def box_constraints(x: Real, y: Real, r: float, cds: List[(float, float)]):
    return Or([
        Or(
            And(x <= cx + r, x >= cx - r, y == cy - r),
            And(x <= cx + r, x >= cx - r, y == cy + r),
            And(y <= cy + r, y >= cy - r, x == cx - r),
            And(y <= cy + r, y >= cy - r, x == cx + r)
        )
        for (cx, cy) in cds
    ])


def extract_model(o: Optimize, x: Real, y: Real, x_0: float, y_0: float):
    d = []
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
    else:
        d.append(-1)
    return d


def observe(agent: Agent, env: Environment):
    cds = [(a.x, agent.y) for a in env.agents if a != agent]
    (x_0, y_0) = (agent.x, agent.y)

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

    range_constraint = [If(y_rng >= 0, y_rng, - y_rng) - vd < 0,
                        If(x_rng >= 0, x_rng, - x_rng) - vd < 0]
    agent_boxes_constraint = box_constraints(x, y, r, cds)

    distances = []
    for lconstr in [y < y_0, y >= y_0]:
        for a in np.linspace(0, np.pi, 7):
            o = Optimize()
            o.add(
                And(
                    # pencil of lines (set of lines passing through a common point):
                    #   (x - x_0) * sin(a) = (y - y_0) * cos(a) forall a in [0, pi]
                    (x - x_0) * np.sin(a) - (y - y_0) * np.cos(a) == 0,
                    # visual depth
                    And(range_constraint),
                    # agent' boxes
                    agent_boxes_constraint,
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
            distances.append(extract_model(o, x, y, x_0, y_0))

    print(distances)
    return distances
