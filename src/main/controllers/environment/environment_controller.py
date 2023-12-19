from typing import List

import numpy as np
import tensorflow as tf

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.learner.learner import Learner
from src.main.model.agents.agent_type import AgentType
from src.main.model.environment.buffer.buffer import Buffer
from src.main.model.environment.environment import Environment


class EnvironmentController:

    def __init__(self, environment: Environment, agent_controllers: List[AgentController],
                 buffers: List[Buffer], learners: List[Learner]):
        self.environment = environment
        self.max_acc = 0.2
        self.t_step = 1
        self.agent_controllers = agent_controllers
        self.buffers = buffers
        self.learners = learners
        self.total_iterations = 50_000

    def train(self):
        """
        Starts the training
        :return:
        """
        prev_states = self._states()
        for it in range(self.total_iterations):
            # avg_rewards = {agent.id: 0 for agent in self.environment.agents}
            for k in range(20):
                # Collect all agents action
                actions = self._actions(prev_states)
                # Move all the agents at once and get their rewards only after
                next_states = self._step(actions)
                rewards = self._rewards()
                print(next_states)
                print(rewards)
                # print([reward for reward in rewards])
                for i, agent_type in enumerate(AgentType):
                    self._record_by_type(agent_type, self.buffers[i], prev_states, actions, rewards, next_states)

            # print([(p_id, r / 10) for p_id, r in avg_rewards.items()])
            for learner in self.learners:
                learner.update()

    def _states(self):
        """
        Gets each agent current state.
        :return: the joint state, a dict of key: agent_id, value: state
        """
        return {
            agent_controller.agent.id: agent_controller.state(self.environment.agents)
            for agent_controller in self.agent_controllers
        }

    def _actions(self, states):
        """
        Gets each agent action based on its current state.
        :param states: joint state
        :return: the joint action, a dict of key: agent_id, value: action
        """
        actions = {}
        for agent_controller in self.agent_controllers:
            agent = agent_controller.agent
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(states[agent.id].distances), 0)
            action = agent_controller.policy(tf_prev_state)
            actions.update({agent.id: list(action)})
        return actions

    def _step(self, actions):
        """
        Moves each agent of one step, returning the new joint state.
        :param actions: joint action
        :return: joint state
        """
        for (agent_id, action) in actions.items():
            self._step_agent(agent_id, action)
        return self._states()

    def _rewards(self):
        """
        Gets each agent reward.
        :return: a dict of key: agent_id, value: reward
        """
        return {
            agent_controller.agent.id: agent_controller.reward()
            for agent_controller in self.agent_controllers
        }

    def _record_by_type(self, agent_type: AgentType,
                        buffer: Buffer,
                        prev_states, actions, rewards, next_states
                        ):
        """
        Records inside the buffer given as parameter the observation tuple of the agents,
        where each agent is of a given type.
        :param agent_type: agent type
        :param buffer: buffer
        :param prev_states: joint state
        :param actions: joint action
        :param rewards: joint rewards
        :param next_states: joint next states
        :return:
        """
        prev_states_t, actions_t, rewards_t, next_states_t = [], [], [], []
        agents = [agent_controller.agent for agent_controller in self.agent_controllers if
                  agent_controller.agent.agent_type == agent_type]
        # avg_rewards = {}
        for agent in agents:
            prev_states_t += prev_states[agent.id].distances
            actions_t += actions[agent.id]
            rewards_t.append(rewards[agent.id])
            next_states_t += next_states[agent.id].distances
            # avg_rewards.update({agent.id: avg_rewards[agent.id] + rewards_t[agent.id]})
        # print(avg_rewards)
        buffer.record((prev_states_t, actions_t, rewards_t, next_states_t))

    def _get_agent_by_id(self, agent_id: str):
        """
        Get the agent with the specified id.
        :param agent_id: agent id to search for
        :return: agent if any, None otherwise
        """
        return next((agent for agent in self.environment.agents if agent.id == agent_id), None)

    def _step_agent(self, agent_id, action):
        """
        Step an agent inside the environment given its action
        :param agent_id: id of the agent
        :param action: respective action
        :return:
        """
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
