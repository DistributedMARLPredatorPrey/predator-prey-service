from typing import List, Dict

import numpy as np
import tensorflow as tf

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.learner.learner import Learner
from src.main.model.agents.agent import Agent
from src.main.model.agents.agent_type import AgentType
from src.main.model.environment.buffer.buffer import Buffer
from src.main.model.environment.environment import Environment
from src.main.model.environment.state import State


class EnvironmentController:

    def __init__(self, environment: Environment, agent_controllers: List[AgentController],
                 buffers: List[Buffer], learners: List[Learner]):
        self.environment = environment
        self.max_acc = 0.2
        self.t_step = 1
        self.agent_controllers = agent_controllers
        self.buffers = buffers
        self.learners = learners

    def train(self):
        """
        Starts the training
        :return:
        """
        # Initial states
        prev_obs_dict = {}
        for agent_controller in self.agent_controllers:
            prev_obs_dict.update({agent_controller.agent.id: agent_controller.state(self.environment.agents)})

        # Train
        total_iterations = 50_000
        for it in range(total_iterations):

            avg_rewards = {}
            for agent in self.environment.agents:
                avg_rewards.update({agent.id: 0})

            for k in range(5):

                # Get the actions from the agents
                actions_dict = {}
                for agent_controller in self.agent_controllers:
                    agent_id = agent_controller.agent.id
                    tf_prev_state = tf.expand_dims(
                        tf.convert_to_tensor(prev_obs_dict[agent_id].state), 0
                    )
                    action = agent_controller.policy(tf_prev_state)
                    actions_dict.update({agent_id: list(action)})

                # Move all the agents at once and get their rewards only after
                next_obs_dict = self._step(actions_dict)
                rewards_dict = self._rewards()

                agents_by_type = [[agent for agent in self.environment.agents
                                   if agent.agent_type == AgentType.PREDATOR],
                                  [agent for agent in self.environment.agents
                                   if agent.agent_type == AgentType.PREY]]

                for i in range(len(agents_by_type)):

                    prev_obs, actions, rewards, next_obs = [], [], [], []
                    for agent in agents_by_type[i]:
                        agent_id = agent.id
                        prev_obs += prev_obs_dict[agent_id].state
                        actions += actions_dict[agent_id]
                        rewards.append(rewards_dict[agent_id])
                        next_obs += next_obs_dict[agent_id].state

                        avg_rewards.update({agent_id: avg_rewards[agent_id] + rewards_dict[agent_id]})

                    print([reward for reward in rewards])

                    # print([(a.agent.x, a.agent.y) for a in self.agent_controllers])
                    # Store on the buffer the joint data
                    self.buffers[i].record((prev_obs, actions, rewards, next_obs))

                # Filter dead agents
                # self._filter_done()

            # print([(p_id, r / 10) for p_id, r in avg_rewards.items()])
            for learner in self.learners:
                learner.update()

    # def _filter_done(self):
    #     self.agent_controllers = [agent_controller
    #                               for agent_controller in self.agent_controllers
    #                               if not agent_controller.done()
    #                               ]
    #     self.environment.agents = [
    #         agent_controller.agent for agent_controller in self.agent_controllers
    #     ]

    def _step(self, actions: Dict[str, List[float]]) -> Dict[str, State]:
        for (agent_id, action) in actions.items():
            self._step_agent(agent_id, action)
        states = {}
        for agent_controller in self.agent_controllers:
            states.update({agent_controller.agent.id: agent_controller.state(self.environment.agents)})
        return states

    def _rewards(self) -> Dict[str, int]:
        rewards = {}
        for agent_controller in self.agent_controllers:
            rewards.update({agent_controller.agent.id: agent_controller.reward()})
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
