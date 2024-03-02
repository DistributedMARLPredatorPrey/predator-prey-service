from typing import List

from numpy.random import uniform

from src.main.model.config.config import EnvironmentConfig
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.model.agents.predator import Predator
from src.main.controllers.agents.predator_prey.predator.predator_controller import (
    PredatorController,
)


class PredatorControllerFactory:
    @staticmethod
    def create_from_params(
        env_config: EnvironmentConfig, policy_controller: AgentPolicyController
    ) -> List[PredatorController]:
        """
        Creates a list of PredatorControllers from the given parameters.
        :param env_config: EnvironmentParams
        :return: list of PredatorControllers
        """
        predator_controllers = []
        for i in range(env_config.num_predators):
            predator = Predator(
                id=f"predator_{i}",
                x=uniform(0, env_config.x_dim),
                y=uniform(0, env_config.y_dim),
            )
            predator_controllers.append(
                PredatorController(
                    env_config=env_config,
                    predator=predator,
                    policy_controller=policy_controller,
                )
            )
        return predator_controllers
