from random import randint, uniform
from typing import Dict, Tuple

from main.model.agents.agent import Agent
from main.model.agents.agent_type import AgentType
from main.model.environment import Environment
from main.model.agents.predator import Predator
from main.controllers.environment.observation import observe

import numpy as np


class EnvironmentController:

    def __init__(self):
        self.environment = None
        self.num_states = 7
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1
        self.max_acc = 0.1
        self.t_step = 0.4

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
        # if type_no is not None:
        #    i = 0
        #    for key in type_no:
        #        if key == AgentType.PREDATOR:
        #            v = type_no[key]
        #            n = i
        #            for k in range(n, n + v[0]):
        #                agents.append(Predator(p_id=k, x=v[1][0], y=v[1][1]))
        #                i = i + 1
        # else:

        n_predators, n_preys = randint(1, 5), randint(1, 5)
        for i in range(n_predators):
            agents.append(Predator(id="predator_${id}".format(id=i),
                                   x=uniform(0, env_x_dim), y=uniform(0, env_y_dim),
                                   vx=0.2, vy=0.2, theta=0, acc=0))

        self.environment = Environment(x_dim=env_x_dim, y_dim=env_y_dim, agents=agents)
        return self.environment

    # agent action
    def step(self, agent: Agent, action: Tuple[float, float]):
        acc, turn = action

        max_incr = self.max_acc * self.t_step
        v = np.sqrt(np.power(agent.vx, 2) + np.power(agent.vy, 2))
        # Compute the new velocity magnitude from the decided acceleration
        new_v = v + acc * max_incr
        # Compute the new direction
        prev_dir = np.arctan2(agent.vx, agent.vy)
        next_dir = prev_dir - turn
        # Compute vx and vy from |v| and the direction
        agent.vx = new_v * np.cos(next_dir)
        agent.vy = new_v * np.sin(next_dir)
        # Compute the next position of the agent, checking if it is inside the boundaries
        next_x = agent.x + agent.vx * self.t_step
        next_y = agent.y + agent.vy * self.t_step
        if next_x >= 0 or next_x < self.environment.x_dim:
            agent.x = next_x
        if next_y >= 0 or next_y < self.environment.y_dim:
            agent.y = next_y
        return observe(agent, self.environment)
