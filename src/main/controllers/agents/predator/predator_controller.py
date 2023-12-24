import numpy as np

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.environment.params.environment_params import EnvironmentParams


class PredatorController(AgentController):
    def __init__(
        self,
        env_params: EnvironmentParams,
        predator: Predator,
        par_service: ParameterService,
    ):
        super().__init__(env_params, predator, par_service)

    def reward(self) -> float:
        r"""
        The predator reward is an exponential function that gets higher as the distance
        of the closest prey decreases.
        Furthermore, it's a min-max normalized function, so that when the distance is 0 the reward is 1,
        and when it's the maximum visual depth it's equal to 0:

        .. math::
            f(x, d) = \frac{e^{-x} - e^{-d}}{1 - e^{-d}}

        :return: the reward
        """
        return (
            np.power(np.e, -np.min(self.last_state.distances))
            - np.power(np.e, -self.vd)
        ) / (1 - np.power(np.e, -self.vd))

    def done(self) -> bool:
        """
        The predator is done when life is equal to zero.
        :return: true if it is done, false otherwise
        """
        return self.life == 0
