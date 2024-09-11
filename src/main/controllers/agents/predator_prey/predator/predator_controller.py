import logging
from typing import List

import numpy as np

from src.main.model.environment.agents.agent import Agent
from src.main.model.config.config import EnvironmentConfig
from src.main.controllers.agents.policy.agent_policy_controller import (
    AgentPolicyController,
)
from src.main.controllers.agents.agent_controller import AgentController
from src.main.model.environment.agents.predator import Predator


class PredatorController(AgentController):
    def __init__(
        self,
        env_config: EnvironmentConfig,
        predator: Predator,
        policy_controller: AgentPolicyController,
    ):
        super().__init__(env_config, predator, policy_controller)

    def reward(self) -> float:
        r"""
        The predator reward is an exponential function that gets higher as the distance
        of the closest prey decreases.
        Furthermore, it's a min-max normalized function, so that when the distance is 0 the reward is 1,
        and when it's the maximum visual depth it's equal to 0:

        .. math::
            f(x, d) = \frac{e^{-x} - e^{-d}}{1 - e^{-d}}

        :return:  the reward
        """
        # return (
        #     np.power(np.e, -np.min(self.last_state.distances)) - np.power(np.e, -1)
        # ) / (1 - np.power(np.e, -1)) * 1000 - 1000
        return -np.min(self.last_state.distances) + 1

    def done(self, _: List[Agent]) -> bool:
        """
        The predator is done when life is equal to zero.
        :return: true if it is done, false otherwise
        """
        self.life = self.life - 1
        return self.life == 0
