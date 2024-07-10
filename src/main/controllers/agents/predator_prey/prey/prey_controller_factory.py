from typing import List

from numpy.random import uniform

from src.main.model.config.config import EnvironmentConfig
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.model.agents.prey import Prey
from src.main.controllers.agents.predator_prey.prey.prey_controller import (
    PreyController,
)


class PreyControllerFactory:
    @staticmethod
    def create_from_params(
        env_config: EnvironmentConfig, policy_controller: AgentPolicyController
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
                x=uniform(0, env_config.x_dim),
                y=uniform(0, env_config.y_dim),
                vx=uniform(-1, 1),
                vy=uniform(-1, 1),
            )
            prey_controllers.append(
                PreyController(
                    env_config=env_config,
                    prey=prey,
                    policy_controller=policy_controller,
                )
            )
        return prey_controllers
