from random import randint, uniform
from typing import Dict, Tuple

from main.model.agents.agent import Agent
from main.model.agents.agent_type import AgentType
from main.model.environment import Environment
from main.model.agents.predator import Predator
from main.controllers.environment.observation import observe

import numpy as np


class EnvironmentController:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.num_states = 7
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1
        self.max_acc = 0.1
        self.t_step = 0.4

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
