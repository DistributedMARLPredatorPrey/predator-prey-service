import numpy as np

from src.main.model.environment.params.environment_params import EnvironmentParams
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.prey import Prey


class PreyController(AgentController):

    def __init__(self, env_params: EnvironmentParams, prey: Prey,
                 par_service: ParameterService):
        super().__init__(env_params, prey, par_service)

    def reward(self) -> float:
        """
        The prey reward is an exponential function that gets higher as the distance
        of the closest predator increases.
        Furthermore, it's a normalized function, so that when the distance is 0 the reward is 0,
        and when it's the maximum visual depth it's equal to 1:
        f(x_min, d_max) = (1 - e^(-x_min)) / (1 - e^(-d_max))
        :return: the reward
        """
        #
        return ((1 - np.power(np.e, -np.min(self.last_state.distances))) /
                (1 - np.power(np.e, -self.vd)))

    def done(self) -> bool:
        """
        :return:
        """
        return False
