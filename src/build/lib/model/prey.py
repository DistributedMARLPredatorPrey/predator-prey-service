from main.model import Agent
from main.model import AgentType


class Prey(Agent):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.agent_type = AgentType.PREY
