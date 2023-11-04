from model.agent_type import AgentType


class Agent:
    def __init__(self, id=None, x=None, y=None, v=0, acc=0, agent_type: AgentType = None):
        self.id = id
        self.x = x
        self.y = y
        self.v = v
        self.acc = acc
        self.agent_type = agent_type
