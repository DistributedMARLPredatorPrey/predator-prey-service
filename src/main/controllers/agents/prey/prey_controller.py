import numpy as np

from src.main.model.environment.params.environment_params import EnvironmentParams
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.prey import Prey


class PreyController(AgentController):

    def __init__(self, env_params: EnvironmentParams, prey: Prey, par_service: ParameterService):
        self.is_done = False
        super().__init__(env_params, prey, par_service)

    def reward(self):
        return np.min(self.last_obs.observation)

    def done(self):
        return self.is_done
