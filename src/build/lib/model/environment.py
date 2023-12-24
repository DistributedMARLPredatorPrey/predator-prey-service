from typing import List

from src.main.model.agents.agent import Agent


class Environment:

    def __init__(self, x_dim=None, y_dim=None, agents: List[Agent] = None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.agents = agents
