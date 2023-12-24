from main.model import AgentType


class Agent:
    def __init__(self, id=None, x=None, y=None, agent_type: AgentType = None):
        self.id = id
        self.x = x
        self.y = y
        self.agent_type = agent_type
