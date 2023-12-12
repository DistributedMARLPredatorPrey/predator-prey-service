from typing import List

from src.main.model.agents.agent_type import AgentType
from src.main.model.agents.agent import Agent
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator


class PredatorController(AgentController):

    def __init__(self, lower_bound: float, upper_bound: float, r: float, life: int,
                 predator: Predator,
                 par_service: ParameterService
                 ):
        self.life = life
        super().__init__(lower_bound, upper_bound, r, predator, par_service)

    def reward(self, agents: List[Agent]):
        num_preys_eaten = len([target for target in agents
                               if target != self.agent and
                               target.agent_type == AgentType.PREY and
                               self.eat(target)
                               ])
        if num_preys_eaten == 0:
            self.life = self.life - 1
            return -1
        else:
            return num_preys_eaten

    def done(self):
        return self.life == 0
