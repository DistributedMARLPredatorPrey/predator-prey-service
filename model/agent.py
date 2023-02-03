from model.atype import AgentType


class Agent:
    def __init__(self, id=None, x=None, y=None, agent_type: AgentType = None):
        self.id = id
        self.x = x
        self.y = y
        self.agent_type = agent_type

    def set_position(self, x=None, y=None):
        self.x = x
        self.y = y
