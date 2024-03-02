from typing import List

import numpy as np
import tensorflow as tf
from z3 import Or, And, If, Solver, Optimize, AlgebraicNumRef, sat, Real

from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.model.environment.params.environment_params import EnvironmentParams
from src.main.model.agents.agent import Agent
from src.main.model.environment.state import State


class AgentController:
    def __init__(
        self, env_params: EnvironmentParams, agent: Agent,
            policy_controller: AgentPolicyController
    ):
        self.last_state = None
        self.num_states = env_params.num_states
        self.lower_bound = env_params.lower_bound
        self.upper_bound = env_params.upper_bound
        self.life = env_params.life
        self.r = env_params.r
        self.vd = env_params.vd
        self.agent = agent
        self.policy_controller = policy_controller

    def action(self, state, verbose=False):
        """
        Computes the next action based on the current state, by getting the current actor model
        from the parameter server.
        :param state: current state
        :param verbose: default set to False
        :return: the next action to be taken
        """
        # the policy used for training just add noise to the action
        # the amount of noise is kept constant during training
        sampled_action = tf.squeeze(self.policy_controller.policy(state))
        noise = np.random.normal(scale=0.1, size=2)

        # we may change the amount of noise for actions during training
        noise[0] *= 2
        noise[1] *= 0.5

        # Adding noise to action
        sampled_action = sampled_action.numpy()
        sampled_action += noise

        # in verbose mode, we may print information about selected actions
        if verbose and sampled_action[0] < 0:
            print("decelerating")

        # Finally, we ensure actions are within bounds
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(legal_action)

    def is_eaten(self, target: Agent) -> bool:
        """
        Checks if this agent is eaten by the target agent given as parameter
        :param target: target agent
        :return: True if the current agent is being eaten, False otherwise
        """
        x, y = Real("x"), Real("y")
        s = Solver()
        s.add(
            x < self.agent.x + self.r,
            x >= self.agent.x - self.r,
            x < target.x + self.r,
            x >= target.x - self.r,
            y < self.agent.y + self.r,
            y >= self.agent.y - self.r,
            y < target.y + self.r,
            y >= target.y - self.r,
        )
        return s.check() == sat

    def state(self, agents: List[Agent]) -> State:
        r"""
        Captures the state given the other agents inside the environment.
        A state is view of the surrounding area, with a given visual depth.

        More specifically it finds the intersection points given these equations and constraints:

        - Pencil of lines (set of lines passing through a common point)

            .. math:: (x - x_0) \sin{a} = (y - y_0) * \cos{a} \quad \forall a \in [0, pi]

        - Constraint x and y to the maximum visual depth:

            .. math:: |y - y_0| < vd, |x - x_0| < vd

        - Box of center (x_c, y_c) and radius r:

            .. math:: x_c - r \leq x \leq x_c + r, y_c - r \leq y \leq y_c + r

        :param agents: other agents inside the environment
        :return: a new state
        """
        cds = np.array(
            [
                (agent.x, agent.y)
                for agent in agents
                if agent != self.agent and agent.agent_type != self.agent
            ]
        )
        (x_0, y_0) = (self.agent.x, self.agent.y)

        x, y = Real("x"), Real("y")
        y_rng = y - y_0
        x_rng = x - x_0

        range_constraint = [
            If(y_rng > 0, y_rng, If(y_rng < 0, -y_rng, self.vd - 1)) - self.vd < 0,
            If(x_rng > 0, x_rng, If(x_rng < 0, -x_rng, self.vd - 1)) - self.vd < 0,
        ]

        agent_boxes_constraint = self._box_constraints(x, y, cds)
        distances = []
        for a in np.linspace(0, np.pi, int(self.num_states / 2), endpoint=False):
            half_line_constraints = (
                [y > y_0, y < y_0] if a != 0 else [x >= x_0, x < x_0]
            )
            for half_line_constraint in half_line_constraints:
                o = Optimize()
                o.add(
                    And(
                        (x - x_0) * np.sin(a) - (y - y_0) * np.cos(a) == 0,
                        And(range_constraint),
                        agent_boxes_constraint,
                        half_line_constraint,
                    )
                )
                o.minimize(If(y > y_0, y, If(y < y_0, -y, If(x >= x_0, x, -x))))
                distances.append(self._extract_distance(o, x, y, x_0, y_0))

        self.last_state = State(distances)
        return self.last_state

    def reward(self) -> float:
        """
        Base reward method, to be overridden by subclasses
        :return: reward as a float number
        """
        raise NotImplementedError("Subclasses must implement this method")

    def done(self) -> bool:
        """
        Base done method, to be overridden by subclasses
        :return: True if it is done, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _box_constraints(self, x: Real, y: Real, cds):
        """
        Box constraints ensure that the intersection point lies in the side of the box
        :param x: symbolic target variable of the x-coordinate
        :param y: symbolic target variable of the y-coordinate
        :param cds: other agents positions
        :return: an Or encoding the mentioned constraints
        """
        return Or(
            [
                Or(
                    And(x <= cx + self.r, x >= cx - self.r, y == cy - self.r),
                    And(x <= cx + self.r, x >= cx - self.r, y == cy + self.r),
                    And(y <= cy + self.r, y >= cy - self.r, x == cx - self.r),
                    And(y <= cy + self.r, y >= cy - self.r, x == cx + self.r),
                )
                for (cx, cy) in cds
            ]
        )

    def _extract_distance(self, o: Optimize, x: Real, y: Real, x_0: float, y_0: float):
        """
        Checks if the Optimize object and extract distance, if SAT.
        :param o: Optimize object to check its satisfiability
        :param x: x-coordinate variable to evaluate
        :param y: y-coordinate variable to evaluate
        :param x_0: x-coordinate of the reference agent
        :param y_0: y-coordinate of the reference agent
        :return: distance, if SAT
        """
        if o.check() == sat:
            model = o.model()

            mx, my = model[x], model[y]
            if isinstance(mx, AlgebraicNumRef):
                mx = mx.approx(10)
            if isinstance(my, AlgebraicNumRef):
                my = my.approx(10)

            x_p, y_p = (
                float(mx.numerator_as_long()) / float(mx.denominator_as_long()),
                float(my.numerator_as_long()) / float(my.denominator_as_long()),
            )
            # Compute the distance between the agent center (x_0, y_0)
            return np.sqrt(np.power(x_0 - x_p, 2) + np.power(y_0 - y_p, 2))
        return self.vd
