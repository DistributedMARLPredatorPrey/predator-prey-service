import logging
from typing import List

import numpy as np
import tensorflow as tf

from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.environment.utils.environment_controller_utils import (
    EnvironmentControllerUtils,
)
from src.main.controllers.replay_buffer.remote.remote_replay_buffer_controller import (
    RemoteReplayBufferController,
)
from src.main.model.environment.agents.agent_type import AgentType
from src.main.model.environment.environment import Environment


class EnvironmentController:
    def __init__(
        self,
        environment: Environment,
        agent_controllers: List[AgentController],
        buffer_controller: RemoteReplayBufferController,
        policy_controllers: List[AgentPolicyController],
        env_controller_utils: EnvironmentControllerUtils,
    ):
        self.environment = environment
        self.max_acc = 0.5
        self.t_step = 2
        self.agent_controllers = agent_controllers
        self.buffer_controller = buffer_controller
        self.policy_controllers = policy_controllers
        self.utils = env_controller_utils

    def train(self):
        """
        Starts the training
        :return:
        """
        prev_states = self.__states()
        while not self.__is_done():
            # Collect all agents action
            actions = self.__actions(prev_states)
            # Move all the agents at once and get their rewards only after
            next_states, rewards = self.__step(actions), self.__rewards()
            logging.info([(ac.agent.x, ac.agent.y) for ac in self.agent_controllers])
            self.__record_to_buffer(prev_states, actions, rewards, next_states)
            prev_states = next_states
        self.__stop_policy_controllers()

    def __states(self):
        """
        Gets each agent current state.
        :return: the joint state, a dict of key: agent_id, value: state
        """
        return {
            agent_controller.agent.id: agent_controller.state(self.environment.agents)
            for agent_controller in self.agent_controllers
        }

    def __actions(self, states):
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

    def __stop_policy_controllers(self):
        for policy_controller in self.policy_controllers:
            policy_controller.stop()

    def __is_done(self):
        return any(
            [
                ac.done(
                    self.__agent_by_type()[
                        AgentType.PREDATOR
                        if ac.agent.agent_type == AgentType.PREY
                        else AgentType.PREY
                    ]
                )
                for ac in self.agent_controllers
            ]
        )

    def __step(self, actions):
        """
        Moves each agent of one step, returning the new joint state.
        :param actions: joint action
        :return: joint state
        """
        for agent_id, action in actions.items():
            self.__step_agent(agent_id, action)
        return self.__states()

    def __rewards(self):
        """
        Gets each agent reward.
        :return: a dict of key: agent_id, value: reward
        """
        return {
            agent_controller.agent.id: agent_controller.reward()
            for agent_controller in self.agent_controllers
        }

    def __record_to_buffer(self, prev_states, actions, rewards, next_states):
        """
        Records inside the replay_buffer given as parameter the observation tuple of the agents,
        where each agent is of a given type.
        :param prev_states: joint state
        :param actions: joint action
        :param rewards: joint rewards
        :param next_states: joint next states
        :return:
        """
        prev_states_t, actions_t, rewards_t, next_states_t = [], [], [], []
        for agent_controller in self.agent_controllers:
            agent = agent_controller.agent
            prev_states_t.append(prev_states[agent.id].distances)
            actions_t.append(actions[agent.id])
            rewards_t.append(rewards[agent.id])
            next_states_t.append(next_states[agent.id].distances)

        average_rewards = np.average(rewards_t)
        self.utils.save_data(
            average_rewards, [(ac.agent.x, ac.agent.y) for ac in self.agent_controllers]
        )
        logging.info(f"Avg reward: {average_rewards}")
        record_tuple = (prev_states_t, actions_t, rewards_t, next_states_t)
        self.buffer_controller.record(record_tuple)

    def __agent_by_type(self):
        return {
            agent_type: [
                agent_controller.agent
                for agent_controller in self.agent_controllers
                if agent_controller.agent.agent_type == agent_type
            ]
            for agent_type in list(AgentType)
        }

    def __agent_by_id(self, agent_id: str):
        """
        Get the agent with the specified id.
        :param agent_id: agent id to search for
        :return: agent if any, None otherwise
        """
        return next(
            (agent for agent in self.environment.agents if agent.id == agent_id), None
        )

    def __step_agent(self, agent_id, action):
        """
        Step an agent inside the environment given its action
        :param agent_id: id of the agent
        :param action: respective action
        :return:
        """
        agent = self.__agent_by_id(agent_id)
        acc, turn = action[0], action[1]
        max_incr = self.max_acc * self.t_step
        v = np.sqrt(np.square(agent.vx) + np.square(agent.vy))
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
        else:
            agent.x = agent.x - agent.vx * self.t_step
        if 0 <= next_y < self.environment.y_dim:
            agent.y = next_y
        else:
            agent.y = agent.y - agent.vy * self.t_step
