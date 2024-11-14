from typing import List

import tensorflow as tf
from z3 import Or, And, If, Optimize, sat, Real, Solver

import numpy as np
from src.main.controllers.agents.policy.agent_policy_controller import (
    AgentPolicyController,
)
from src.main.model.config.config import EnvironmentConfig
from src.main.model.environment.agents.agent import Agent


class AgentController:
    def __init__(
        self,
        env_config: EnvironmentConfig,
        agent: Agent,
        policy_controller: AgentPolicyController,
    ):
        self.last_state = None
        self.num_states = env_config.num_states
        self.lower_bound = env_config.acc_lower_bound
        self.upper_bound = env_config.acc_upper_bound
        self.life = env_config.life
        self.r = env_config.r
        self.vd = env_config.vd
        self.agent = agent
        self.policy_controller = policy_controller

    def action(self, state):
        """
        Computes the next action based on the current state, by getting the current actor model
        from the parameter server.
        :param state: current state
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

        v, turn = sampled_action
        new_v, new_turn = (
            np.clip(v * 10, -10, 10),
            np.clip(turn * np.pi, -np.pi, np.pi),
        )
        return np.squeeze([new_v, new_turn])

    def state(self, agents: List[Agent]):
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
                if agent != self.agent and agent.agent_type != self.agent.agent_type
            ]
        )
        (x_0, y_0) = (self.agent.x, self.agent.y)

        x, y = Real("x"), Real("y")
        y_rng = y - y_0
        x_rng = x - x_0

        range_constraint = [
            If(y_rng > 0, y_rng, -y_rng) - self.vd < 0,
            If(x_rng > 0, x_rng, -x_rng) - self.vd < 0,
        ]

        agent_boxes_constraint = self.__box_constraints(x, y, cds)
        distances = []

        angles = np.linspace(0, np.pi, int(self.num_states / 2) + 1, endpoint=False)[1:]
        for a in angles:
            half_line_constraints = [x >= x_0, x < x_0]
            for half_line_constraint in half_line_constraints:
                is_sat = True
                solutions = []
                solutions_coords = []
                while is_sat:
                    s = Solver()
                    s.add(
                        And(
                            (x - x_0) * np.sin(a) - (y - y_0) * np.cos(a) == 0,
                            And(range_constraint),
                            agent_boxes_constraint,
                            half_line_constraint,
                            And(
                                [
                                    And(x != s_x, y != s_y)
                                    for s_x, s_y in solutions_coords
                                ]
                            ),
                        )
                    )
                    is_sat, distance, mx, my = self.__extract_distance(
                        s, x, y, x_0, y_0
                    )
                    if is_sat:
                        solutions_coords.append((mx, my))
                    # o.minimize(If(y > y_0, y, If(y < y_0, -y, If(x >= x_0, x, -x))))
                    solutions.append(distance)
                distances.append(np.min(solutions))

        self.last_state = distances
        return self.last_state

    def reward(self) -> float:
        """
        Base reward method, to be overridden by subclasses
        :return: reward as a float number
        """
        raise NotImplementedError("Subclasses must implement this method")

    def done(self, agents: List[Agent]) -> bool:
        """
        Base done method, to be overridden by subclasses
        :param agents: other agents inside the environment
        :return: True if it is done, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __box_constraints(self, x: Real, y: Real, cds):
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

    def __extract_distance(self, o: Optimize, x: Real, y: Real, x_0: float, y_0: float):
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
            # if isinstance(mx, AlgebraicNumRef):
            #     mx = mx.approx(10)
            # if isinstance(my, AlgebraicNumRef):
            #     my = my.approx(10)

            x_p, y_p = (
                float(mx.numerator_as_long()) / float(mx.denominator_as_long()),
                float(my.numerator_as_long()) / float(my.denominator_as_long()),
            )
            # Compute the l2 distance between the agent center (x_0, y_0)
            d = np.linalg.norm(np.array([x_0, y_0]) - np.array([x_p, y_p]))
            return True, d / self.vd, mx, my
        return False, self.vd / self.vd, None, None
