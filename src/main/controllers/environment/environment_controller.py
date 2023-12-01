from typing import List, Dict

import numpy as np

from src.main.controllers.environment.environment_observer import EnvironmentObserver
from src.main.model.agents.agent import Agent
from src.main.model.environment.environment import Environment
from src.main.model.environment.observation import Observation


class EnvironmentController:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.upper_bound = 1
        self.lower_bound = -1
        self.max_acc = 0.1
        self.t_step = 0.4
        self.environment_observer = EnvironmentObserver()

    def step(self, actions: Dict[str, List[float]]) -> Dict[str, Observation]:
        for (agent_id, action) in actions.items():
            self._step_agent(agent_id, action)
        observations = {}
        for agent_id in actions:
            observations.update({agent_id: self.observe(self._get_agent_by_id(agent_id))})
        return observations

    def rewards(self) -> Dict[str, int]:
        rewards = {}
        for agent in self.environment.agents:
            rewards.update({agent.id: self.environment_observer.reward(agent, self.environment)})
        return rewards

    def observe(self, agent: Agent) -> Observation:
        return self.environment_observer.observe(agent, self.environment)

    def _get_agent_by_id(self, agent_id: str) -> Agent:
        for agent in self.environment.agents:
            if agent.id == agent_id:
                return agent

    # agent action
    def _step_agent(self, agent_id: str, action: List[float]):
        agent = self._get_agent_by_id(agent_id)
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
