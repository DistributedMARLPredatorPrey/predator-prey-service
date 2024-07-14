import random
from typing import List

from src.main.controllers.agents.predator_prey.prey.prey_controller import (
    PreyController,
)
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.model.config.config import EnvironmentConfig
from src.main.model.environment.agents.prey import Prey


class PreyControllerFactory:
    @staticmethod
    def create_from_config(
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
                x=random.uniform(0, env_config.x_dim),
                y=random.uniform(0, env_config.y_dim),
                vx=random.uniform(-1, 1),
                vy=random.uniform(-1, 1),
            )
            prey_controllers.append(
                PreyController(
                    env_config=env_config,
                    prey=prey,
                    policy_controller=policy_controller,
                )
            )
        return prey_controllers
