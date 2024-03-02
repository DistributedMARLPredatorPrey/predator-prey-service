from typing import List

from numpy.random import uniform

from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.model.agents.prey import Prey
from src.main.controllers.agents.predator_prey.prey.prey_controller import PreyController
from src.main.model.environment.params.environment_params import EnvironmentParams


class PreyControllerFactory:
    @staticmethod
    def create_from_params(env_params: EnvironmentParams,
                           policy_controller: AgentPolicyController) -> List[PreyController]:
        """
        Creates a list of PreyControllers from the given parameters.
        :param env_params: EnvironmentParams
        :return: list of PreyControllers
        """
        prey_controllers = []
        for i in range(env_params.num_predators):
            prey = Prey(id=f"prey_{i}", x=uniform(0, env_params.x_dim),
                        y=uniform(0, env_params.y_dim))
            prey_controllers.append(
                PreyController(
                    env_params=env_params, prey=prey, policy_controller=policy_controller
                )
            )
        return prey_controllers
