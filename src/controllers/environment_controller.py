from random import randint, uniform
from typing import Dict, Tuple

from src.model.agents.agent_type import AgentType
from src.model.environment import Environment
from src.model.agents.predator import Predator


class EnvironmentController:
    def __init__(self):
        self.environment = None
        self.num_states = 7
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1

    # def collision(id, id)
    # def observation(id)

    def reset(self, type_no: Dict[AgentType, Tuple[int, Tuple[float, float]]] = None,
              x_dim=None, y_dim=None):

        """
            Resets the environment by deleting all the current agents
            and inserting new ones

            :param type_no: Dict[AgentType, (agent_id, (x, y))]. For each agent type \
            a number of respected agents are passed as input
            :param x_dim: environment's x-axis dimension
            :param y_dim: environment's y-axis dimension
            :type type_no: Dict[AgentType, (int, (float, float))] or None
            :type x_dim: int or None
            :type y_dim: int or None
            :return: The environment
            :rtype: Environment
        """

        env_x_dim, env_y_dim = (500, 500) \
            if (x_dim is None or y_dim is None) else (x_dim, y_dim)
        agents = []
        if type_no is not None:
            i = 0
            for key in type_no:
                if key == AgentType.PREDATOR:
                    v = type_no[key]
                    n = i
                    for k in range(n, n + v[0]):
                        agents.append(Predator(id=k, x=v[1][0], y=v[1][1]))
                        i = i + 1

        else:
            n_predators, n_preys = randint(1, 5), randint(1, 5)
            for i in range(n_predators):
                agents.append(Predator(id=i, x=uniform(0, env_x_dim), y=uniform(0, env_y_dim)))

        self.environment = Environment(x_dim=env_x_dim, y_dim=env_y_dim, agents=agents)
        return self.environment

    # We introduce a probability of doing n empty actions to separate the environment time-step from the agent
    # def step(self, action)
