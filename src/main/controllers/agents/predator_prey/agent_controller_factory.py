from typing import List

import numpy as np

from src.main.model.environment.agents.predator import Predator
from src.main.controllers.agents.predator_prey.predator.predator_controller import (
    PredatorController,
)
from src.main.controllers.agents.predator_prey.prey.prey_controller import (
    PreyController,
)
from src.main.controllers.agents.policy.agent_policy_controller import (
    AgentPolicyController,
)
from src.main.model.config.config import EnvironmentConfig
from src.main.model.environment.agents.prey import Prey


class AgentControllerFactory:
    @staticmethod
    def predator_controllers_from_config(
        env_config: EnvironmentConfig,
        policy_controller: AgentPolicyController,
    ) -> List[PredatorController]:
        """
        Creates a list of PredatorControllers from the given parameters.
        :param policy_controller: AgentPolicyController
        :param env_config: EnvironmentParams
        :return: list of PredatorControllers
        """
        predator_controllers = []
        for i in range(env_config.num_predators):
            predator = Predator(
                id=f"predator_{i}",
                x=np.random.uniform(0, env_config.x_dim),
                y=np.random.uniform(0, env_config.y_dim),
                vx=np.random.uniform(0, 10),
                vy=np.random.uniform(0, 10),
            )
            predator_controllers.append(
                PredatorController(
                    env_config=env_config,
                    predator=predator,
                    policy_controller=policy_controller,
                )
            )
        return predator_controllers

    @staticmethod
    def prey_controllers_from_config(
        env_config: EnvironmentConfig,
        policy_controller: AgentPolicyController,
    ) -> List[PreyController]:
        """
        Creates a list of PreyControllers from the given parameters.
        :param policy_controller: AgentPolicyController
        :param env_config: EnvironmentConfig
        :return: list of PreyControllers
        """
        prey_controllers = []
        for i in range(env_config.num_preys):
            prey = Prey(
                id=f"prey_{i}",
                x=np.random.uniform(0, env_config.x_dim),
                y=np.random.uniform(0, env_config.y_dim),
                vx=np.random.uniform(0, 10),
                vy=np.random.uniform(0, 10),
            )
            prey_controllers.append(
                PreyController(
                    env_config=env_config,
                    prey=prey,
                    policy_controller=policy_controller,
                )
            )
        return prey_controllers
