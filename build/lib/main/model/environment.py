from typing import List

from src.main.model.agents.agent import Agent


class Environment:
    def __init__(self, x_dim: int = 500, y_dim: int = 500, agents: List[Agent] = None):
        if agents is None:
            agents = []
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.agents = agents
        self.num_states = 14
        self.num_actions = 2
