import numpy as np
import tensorflow as tf

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator


class PredatorController(AgentController):

    def __init__(self, lower_bound: float, upper_bound: float,
                 predator: Predator,
                 par_service: ParameterService
                 ):
        super().__init__(lower_bound, upper_bound, predator, par_service)

    def reward(self, observation):
        pass
