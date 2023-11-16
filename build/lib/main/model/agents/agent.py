from src.main.model.agents.agent_type import AgentType


class Agent:
    def __init__(self, id=None, x=None, y=None, vx=1, vy=1, theta=0, acc=0.5, agent_type: AgentType = None):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.acc = acc
        self.agent_type = agent_type
