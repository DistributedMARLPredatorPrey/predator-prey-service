import numpy as np

from src.main.model.environment.params.environment_params import EnvironmentParams
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.prey import Prey


class PreyController(AgentController):

    def __init__(self, env_params: EnvironmentParams, prey: Prey, par_service: ParameterService):
        super().__init__(env_params, prey, par_service)

    def reward(self) -> float:
        """
        The prey reward is proportional to the distance of the closest predator.
        :return: the reward
        """
        return np.min(self.last_obs.state)

    def done(self) -> bool:
        """
        :return:
        """
        return False
