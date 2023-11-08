from main.model.agents.agent import Agent


class Environment:
    def __init__(self, x_dim=500, y_dim=500, agents=None):
        if agents is None:
            agents = []
        if agents is None:
            agents = []
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.agents = agents
        self.num_states = 14
        self.num_actions = 2
