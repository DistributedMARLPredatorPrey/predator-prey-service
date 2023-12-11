from typing import List, Dict

import numpy as np
import tensorflow as tf

from src.main.model.agents.agent_type import AgentType
from src.main.controllers.agents.buffer import Buffer
from src.main.controllers.agents.predator_controller import PredatorController
from src.main.controllers.environment.environment_observer import EnvironmentObserver
from src.main.controllers.learner.learner import Learner
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.agent import Agent
from src.main.model.environment.environment import Environment
from src.main.model.environment.observation import Observation


class EnvironmentController:

    def __init__(self, environment: Environment):
        # Parameters
        self.environment = environment
        self.upper_bound = 1
        self.lower_bound = -1
        self.max_acc = 0.2
        self.t_step = 1
        self.env_obs = EnvironmentObserver()

        # ParameterService & Learner
        self.par_services = [ParameterService() for _ in range(len(environment.agents))]

        # Predator Controllers
        self.agent_controllers = [
            PredatorController(
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                r=10,
                life=100,
                predator=agent,
                par_service=self.par_services[self.environment.agents.index(agent)]
            )
            for agent in self.environment.agents if agent.agent_type == AgentType.PREDATOR
        ]
        # Buffer
        self.buffer = Buffer(50_000, 64,
                             environment.num_states,
                             environment.num_actions,
                             len(self.agent_controllers)
                             )
        # Learners
        self.learners = [Learner(self.buffer,
                                 self.par_services,
                                 environment.num_states,
                                 environment.num_actions,
                                 len(self.agent_controllers)
                                 )
                         ]

    def train(self):
        # Initial observation
        prev_obs_dict = {}
        for agent in self.environment.agents:
            prev_obs_dict.update({agent.id: self.env_obs.observe(agent, self.environment)})

        # Train
        total_iterations = 50_000
        for it in range(total_iterations):

            avg_rewards = {}
            for agent in self.environment.agents:
                avg_rewards.update({agent.id: 0})

            for k in range(25):
                # Get the actions from the agents
                actions_dict = {}
                for predator_controller in self.agent_controllers:
                    p_id = predator_controller.agent.id
                    tf_prev_state = tf.expand_dims(
                        tf.convert_to_tensor(prev_obs_dict[p_id].observation), 0
                    )
                    action = predator_controller.policy(tf_prev_state)
                    actions_dict.update({p_id: list(action)})

                # Move all the agents at once and get their rewards only after
                next_obs_dict = self._step(actions_dict)
                rewards_dict = self._rewards()

                prev_obs, actions, rewards, next_obs = [], [], [], []
                for agent in self.environment.agents:
                    p_id = agent.id
                    prev_obs += prev_obs_dict[p_id].observation
                    actions += actions_dict[p_id]
                    rewards.append(rewards_dict[p_id])
                    next_obs += next_obs_dict[p_id].observation

                    avg_rewards.update({p_id: avg_rewards[p_id] + rewards_dict[p_id]})

                print([reward for reward in rewards])
                #print([(a.agent.x, a.agent.y) for a in self.agent_controllers])
                # Store on the buffer the joint data
                self.buffer.record((prev_obs, actions, rewards, next_obs))

            for learner in self.learners:
                print([(p_id, r / 25) for p_id, r in avg_rewards.items()])
                learner.update()

    def _step(self, actions: Dict[str, List[float]]) -> Dict[str, Observation]:
        for (agent_id, action) in actions.items():
            self._step_agent(agent_id, action)
        observations = {}
        for agent_id in actions:
            observations.update(
                {agent_id: self.env_obs.observe(self._get_agent_by_id(agent_id), self.environment)}
            )
        return observations

    def _rewards(self) -> Dict[str, int]:
        rewards = {}
        for agent_controller in self.agent_controllers:
            rewards.update({agent_controller.agent.id: agent_controller.reward(self.environment.agents)})
        return rewards

    def _get_agent_by_id(self, agent_id: str) -> Agent:
        for agent in self.environment.agents:
            if agent.id == agent_id:
                return agent

    # Step an Agent in the Environment given an action
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
