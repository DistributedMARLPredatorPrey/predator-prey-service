from typing import List

from src.main.model.environment.agents.agent import Agent


class Environment:
    def __init__(self, x_dim: int = 500, y_dim: int = 500, agents: List[Agent] = None):
        if agents is None:
            agents = []
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.agents = agents
