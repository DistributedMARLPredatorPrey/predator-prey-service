from src.model.agents.agent import Agent
from src.model.agents.agent_type import AgentType


class Predator(Agent):
    def __init__(self, id=None, x=None, y=None, vx=0, vy=0, theta=0, acc=0):
        super().__init__(id, x, y, vx, vy, theta, acc, AgentType.PREDATOR)
