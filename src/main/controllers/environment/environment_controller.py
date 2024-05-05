from typing import List

import numpy as np
import tensorflow as tf
import os
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.replay_buffer.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.model.agents.agent_type import AgentType
from src.main.model.environment.environment import Environment


class EnvironmentController:
    def __init__(
        self,
        environment: Environment,
        agent_controllers: List[AgentController],
        buffer_controller: ReplayBufferController,
    ):
        self.environment = environment
        self.max_acc = 0.2
        self.t_step = 1
        self.agent_controllers = agent_controllers
        self.buffer_controller = buffer_controller
        self.total_iterations = 50_000

    def train(self):
        """
        Starts the training
        :return:
        """
        prev_states = self._states()
        for it in range(self.total_iterations):
            # Collect all agents action
            actions = self._actions(prev_states)
            # Move all the agents at once and get their rewards only after
            next_states, rewards = self._step(actions), self._rewards()
            avg_rewards = np.average(list(rewards.values()))
            with open(f"/usr/app/config/rewards_{os.environ.get('REL_PATH')}.txt", "a") as f:
                f.write(f"{avg_rewards}\n")
            self._record_to_buffer(prev_states, actions, rewards, next_states)
            prev_states = next_states

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
            tf_prev_state = tf.expand_dims(
                tf.convert_to_tensor(states[agent.id].distances), 0
            )
            action = agent_controller.action(tf_prev_state)
            actions.update({agent.id: list(action)})
        return actions

    def _step(self, actions):
        """
        Moves each agent of one step, returning the new joint state.
        :param actions: joint action
        :return: joint state
        """
        for agent_id, action in actions.items():
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

    def _record_to_buffer(self, prev_states, actions, rewards, next_states):
        """
        Records inside the replay_buffer given as parameter the observation tuple of the agents,
        where each agent is of a given type.
        :param prev_states: joint state
        :param actions: joint action
        :param rewards: joint rewards
        :param next_states: joint next states
        :return:
        """
        agents_by_type = self._agent_by_type()
        record_tuples = {}
        for at, agents in agents_by_type.items():
            # record_tuples = {}
            prev_states_t, actions_t, rewards_t, next_states_t = [], [], [], []
            for agent in agents:
                prev_states_t.append(prev_states[agent.id].distances)
                actions_t.append(actions[agent.id])
                rewards_t.append(rewards[agent.id])
                next_states_t.append(next_states[agent.id].distances)
            print(f"Avg reward {at}: {np.average(rewards_t)}")
            record_tuples.update(
                {at: (prev_states_t, actions_t, rewards_t, next_states_t)}
            )
        print("Post data to the replay buffer:", record_tuples)
        self.buffer_controller.record(record_tuples)

    def _agent_by_type(self):
        return {
            agent_type: [
                agent_controller.agent
                for agent_controller in self.agent_controllers
                if agent_controller.agent.agent_type == agent_type
            ]
            for agent_type in list(AgentType)
        }

    def _agent_by_id(self, agent_id: str):
        """
        Get the agent with the specified id.
        :param agent_id: agent id to search for
        :return: agent if any, None otherwise
        """
        return next(
            (agent for agent in self.environment.agents if agent.id == agent_id), None
        )

    def _step_agent(self, agent_id, action):
        """
        Step an agent inside the environment given its action
        :param agent_id: id of the agent
        :param action: respective action
        :return:
        """
        agent = self._agent_by_id(agent_id)
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
