from typing import List

from numpy.random import uniform

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.prey import Prey
from src.main.controllers.agents.prey.prey_controller import PreyController
from src.main.model.environment.params.environment_params import EnvironmentParams


class PreyControllerFactory:

    @staticmethod
    def create_from_params(env_params: EnvironmentParams) -> List[PreyController]:
        """
        Creates a list of PreyControllers from the given parameters.
        :param env_params: EnvironmentParams
        :return: list of PreyControllers
        """
        prey_controllers = []
        for i in range(env_params.num_predators):
            prey = Prey(id=f"prey_{i}", x=uniform(0, env_params.x_dim),
                        y=uniform(0, env_params.y_dim))
            par_service = ParameterService()
            prey_controllers.append(
                PreyController(lower_bound=env_params.lower_bound, upper_bound=env_params.upper_bound,
                               r=env_params.r, life=env_params.life, prey=prey, par_service=par_service
                               )
            )
        return prey_controllers
