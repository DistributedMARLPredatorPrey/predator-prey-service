from typing import List

import numpy as np

from z3 import Solver, sat, Real

from main.model.environment.agents.agent import Agent
from main.model.config.config import EnvironmentConfig
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.controllers.agents.agent_controller import AgentController
from main.model.environment.agents import Prey


class PreyController(AgentController):
    def __init__(
        self,
        env_config: EnvironmentConfig,
        prey: Prey,
        policy_controller: AgentPolicyController,
    ):
        super().__init__(env_config, prey, policy_controller)

    def reward(self) -> float:
        r"""
        The prey reward is an exponential function that gets higher as the distance
        of the closest predator increases.
        Furthermore, it's a min-max normalized function, so that when the distance is 0 the reward is 0,
        and when it's the maximum visual depth it's equal to 1:

        .. math::
            f(x, d) = \frac{1 - e^{-x}}{1- e^{-d}}

        :return: the reward
        """
        return (1 - np.power(np.e, -np.min(self.last_state.distances))) / (
            1 - np.power(np.e, -self.vd)
        ) - 1

    def done(self, agents: List[Agent]) -> bool:
        """
        Checks if this agent is eaten by the target agents given as parameter
        :param agents: other agents inside the environment
        :return: True if the current agent is being eaten, False otherwise
        """
        for agent in agents:
            x, y = Real("x"), Real("y")
            s = Solver()
            s.add(
                x < self.agent.x + self.r,
                x >= self.agent.x - self.r,
                x < agent.x + self.r,
                x >= agent.x - self.r,
                y < self.agent.y + self.r,
                y >= self.agent.y - self.r,
                y < agent.y + self.r,
                y >= agent.y - self.r,
            )
            if s.check() == sat:
                return True
        return False
