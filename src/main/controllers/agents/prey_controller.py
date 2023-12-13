import numpy as np

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.prey import Prey


class PreyController(AgentController):

    def __init__(self, lower_bound: float, upper_bound: float, r: float, life: int,
                 prey: Prey,
                 par_service: ParameterService
                 ):
        self.is_done = False
        super().__init__(lower_bound, upper_bound, r, prey, par_service)

    def reward(self):
        return np.min(self.last_obs.observation)

    def done(self):
        return self.is_done
