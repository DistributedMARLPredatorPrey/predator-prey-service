from typing import List

import numpy as np

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.agent import Agent
from src.main.model.agents.agent_type import AgentType
from src.main.model.agents.predator import Predator


class PredatorController(AgentController):

    def __init__(self, lower_bound: float, upper_bound: float, r: float, life: int,
                 predator: Predator,
                 par_service: ParameterService
                 ):
        self.life = life
        super().__init__(lower_bound, upper_bound, r, predator, par_service)

    def reward(self):
        return 1 / (1 + np.min(self.last_obs.observation))

    def done(self):
        return self.life == 0
