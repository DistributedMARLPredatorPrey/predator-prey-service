from typing import List

from numpy.random import uniform

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.controllers.agents.predator.predator_controller import PredatorController
from src.main.model.environment.params.environment_params import EnvironmentParams


class PredatorControllerFactory:

    @staticmethod
    def create_from_params(env_params: EnvironmentParams) -> List[PredatorController]:
        """
        Creates a list of PredatorControllers from the given parameters.
        :param env_params: EnvironmentParams
        :return: list of PredatorControllers
        """
        predator_controllers = []
        for i in range(env_params.num_predators):
            predator = Predator(id=f"predator_{i}", x=uniform(0, env_params.x_dim),
                                y=uniform(0, env_params.y_dim))
            par_service = ParameterService()
            predator_controllers.append(PredatorController(env_params, predator=predator, par_service=par_service))
        return predator_controllers
