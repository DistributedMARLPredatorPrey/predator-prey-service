from typing import Tuple, List

import numpy as np

from src.main.controllers.environment.environment_observer import EnvironmentObserver
from src.main.model.agents.agent import Agent
from src.main.model.environment import Environment


class EnvironmentController:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.upper_bound = 1
        self.lower_bound = -1
        self.max_acc = 0.1
        self.t_step = 0.4

    # agent action
    def step(self, agent: Agent, action: List[float]) -> (List[float], bool, int):
        acc, turn = action[0], action[1]
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
        if 0 <= next_x < self.environment.x_dim:
            agent.x = next_x
        if 0 <= next_y < self.environment.y_dim:
            agent.y = next_y
        return self.observe(agent)

    def observe(self, agent: Agent) -> (List[float], bool, int):
        return EnvironmentObserver().observe(agent, self.environment)
