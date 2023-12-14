from typing import List

import numpy as np
import tensorflow as tf
from z3 import Or, And, If, Solver, Optimize, AlgebraicNumRef, sat, Real

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.agent import Agent
from src.main.model.environment.observation import Observation


class AgentController:

    def __init__(self, lower_bound: float, upper_bound: float, r: float,
                 agent: Agent,
                 par_service: ParameterService):
        self.last_obs = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.r = r
        self.vd = 30
        self.agent = agent
        self.par_service = par_service

    def policy(self, state, verbose=False):
        # the policy used for training just add noise to the action
        # the amount of noise is kept constant during training
        sampled_action = tf.squeeze(self.par_service.actor_model(state))
        noise = np.random.normal(scale=0.1, size=2)

        # we may change the amount of noise for actions during training
        noise[0] *= 2
        noise[1] *= .5

        # Adding noise to action
        sampled_action = sampled_action.numpy()
        sampled_action += noise

        # in verbose mode, we may print information about selected actions
        if verbose and sampled_action[0] < 0:
            print("decelerating")

        # Finally, we ensure actions are within bounds
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(legal_action)

    def eat(self, target: Agent):
        x, y = Real('x'), Real('y')
        s = Solver()
        s.add(x < self.agent.x + self.r, x >= self.agent.x - self.r,
              x < target.x + self.r, x >= target.x - self.r,
              y < self.agent.y + self.r, y >= self.agent.y - self.r,
              y < target.y + self.r, y >= target.y - self.r
              )
        return s.check() == sat

    def observe(self, agents: List[Agent]) -> Observation:
        cds = np.array([(a.x, a.y) for a in agents if a != self.agent and a.agent_type != self.agent])
        (x_0, y_0) = (self.agent.x, self.agent.y)

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

        range_constraint = [If(y_rng >= 0, y_rng, - y_rng) - self.vd < 0,
                            If(x_rng >= 0, x_rng, - x_rng) - self.vd < 0]
        agent_boxes_constraint = self._box_constraints(x, y, self.r, cds)

        distances = []
        for lconstr in [y >= y_0, y < y_0]:
            for a in np.linspace(0, np.pi, 7):
                o = Optimize()
                o.add(
                    And(
                        # pencil of lines (set of lines passing through a common point):
                        # (x - x_0) * sin(a) = (y - y_0) * cos(a) forall a in [0, pi]
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
                distances.extend(self._extract_model(o, x, y, x_0, y_0))
        self.last_obs = Observation(distances)
        return self.last_obs

    @staticmethod
    def _box_constraints(x: Real, y: Real, r: float, cds: List):
        return Or([
            Or(
                And(x <= cx + r, x >= cx - r, y == cy - r),
                And(x <= cx + r, x >= cx - r, y == cy + r),
                And(y <= cy + r, y >= cy - r, x == cx - r),
                And(y <= cy + r, y >= cy - r, x == cx + r)
            )
            for (cx, cy) in cds
        ])

    def _extract_model(self, o: Optimize, x: Real, y: Real, x_0: float, y_0: float):
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
            # print((x_p, y_p))
            # Compute the distance between the agent center (x_0, y_0)
            # and the intersection point which is closer to it
            d.append(round(np.sqrt(np.power(x_0 - x_p, 2) + np.power(y_0 - y_p, 2)), 2))
        else:
            d.append(self.vd)
        return d

    # Base reward method, to be overridden by subclasses
    def reward(self):
        raise NotImplementedError("Subclasses must implement this method")

    # Base done method, to be overridden by subclasses
    def done(self):
        raise NotImplementedError("Subclasses must implement this method")
