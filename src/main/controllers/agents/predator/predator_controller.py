import numpy as np

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.environment.params.environment_params import EnvironmentParams


class PredatorController(AgentController):

    def __init__(self, env_params: EnvironmentParams, predator: Predator,
                 par_service: ParameterService):
        super().__init__(env_params, predator, par_service)

    def reward(self) -> float:
        """
        The predator reward is inversely proportional to the distance of the closest prey.
        :return: reward
        """
        return 1 / (1 + np.min(self.last_obs.distances))

    def done(self) -> bool:
        """
        The predator is done when life is equal to zero.
        :return: true if it is done, false otherwise
        """
        return self.life == 0
