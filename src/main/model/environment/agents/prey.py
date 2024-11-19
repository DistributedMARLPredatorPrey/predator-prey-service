from src.main.model.environment.agents.agent import Agent
from src.main.model.environment.agents.agent_type import AgentType


class Prey(Agent):
    def __init__(self, id=None, x=None, y=None, vx=1, vy=1):
        super().__init__(id, x, y, vx, vy, AgentType.PREY)
