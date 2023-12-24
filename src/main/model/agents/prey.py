from src.main.model.agents.agent import Agent
from src.main.model.agents.agent_type import AgentType


class Prey(Agent):
    def __init__(self, id=None, x=None, y=None, vx=1, vy=1, acc=0.5):
        super().__init__(id, x, y, vx, vy, acc, AgentType.PREY)
