from typing import List

from src.main.model.agents.agent_type import AgentType
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.agent import Agent
from src.main.model.agents.prey import Prey


class PreyController(AgentController):

    def __init__(self, lower_bound: float, upper_bound: float, r: float, life: int,
                 prey: Prey,
                 par_service: ParameterService
                 ):
        self.is_done = False
        super().__init__(lower_bound, upper_bound, r, prey, par_service)

    def reward(self, agents: List[Agent]):
        if any([target for target in agents
                if target != self.agent and
                   target.agent_type == AgentType.PREDATOR and
                   self.eat(target)
                ]):
            self.is_done = True
            return -3
        return 1

    def done(self):
        return self.is_done
