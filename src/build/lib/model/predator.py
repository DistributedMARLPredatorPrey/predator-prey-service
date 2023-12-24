from main.model import Agent
from main.model import AgentType


class Predator(Agent):
    def __init__(self, id=None, x=None, y=None):
        self.id = id
        self.x = x
        self.y = y
        self.agent_type = AgentType.PREDATOR
