from main.model.environment.agents.agent import Agent
from main.model.environment.agents.agent_type import AgentType


class Predator(Agent):
    def __init__(self, id=None, x=None, y=None, vx=1, vy=1, acc=0.5):
        super().__init__(id, x, y, vx, vy, acc, AgentType.PREDATOR)
