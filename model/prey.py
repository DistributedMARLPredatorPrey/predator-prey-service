from model.agent import Agent
from model.atype import AgentType


class Prey(Agent):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.agent_type = AgentType.PREY
