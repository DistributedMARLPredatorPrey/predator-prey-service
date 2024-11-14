import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src.main.model.environment.agents.predator import Predator
from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.environment.utils.environment_controller_utils import (
    EnvironmentControllerUtils,
)
from src.main.controllers.agents.policy.agent_policy_controller import (
    AgentPolicyController,
)
from src.main.controllers.replay_buffer.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.model.environment.agents.agent_type import AgentType
from src.main.model.environment.environment import Environment


class EnvironmentController:
    def __init__(
        self,
        environment: Environment,
        agent_controllers: List[AgentController],
        buffer_controller: ReplayBufferController,
        policy_controllers: List[AgentPolicyController],
        env_controller_utils: EnvironmentControllerUtils,
    ):
        self.__environment = environment
        self.__t_step = 1
        self.__agent_controllers = agent_controllers
        self.__buffer_controller = buffer_controller
        self.__policy_controllers = policy_controllers
        self.__utils = env_controller_utils

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
            # Print and save coords and rewards
            agents_coords = [
                (ac.agent.x, ac.agent.y) for ac in self.__agent_controllers
            ]
            logging.info(agents_coords)
            logging.info(f"Avg reward: {np.average(list(rewards.values()))}")
            self.__utils.save_data(
                list(rewards.values()),  # np.average(list(rewards.values())),
                [(ac.agent.x, ac.agent.y) for ac in self.__agent_controllers],
            )
            # Record to buffer for batch learning
            self.__record_to_buffer((prev_states, actions, rewards, next_states))
            prev_states = next_states
        self.__stop_policy_controllers()

    def simulate(self):
        """
        Starts the simulation
        :return:
        """
        prev_states = self.__states()
        while not self.__is_done():
            actions = self.__actions(prev_states)
            next_states = self.__step(actions)
            logging.info([(ac.agent.x, ac.agent.y) for ac in self.__agent_controllers])
            prev_states = next_states
            self.__utils.save_data(
                [], [(ac.agent.x, ac.agent.y) for ac in self.__agent_controllers]
            )

    def __states(self):
        """
        Gets each agent current state.
        :return: the joint state, a dict of key: agent_id, value: state
        """
        return {
            agent_controller.agent.id: agent_controller.state(self.__environment.agents)
            for agent_controller in self.__agent_controllers
        }

    def __actions(self, states):
        """
        Gets each agent action based on its current state.
        :param states: joint state
        :return: the joint action, a dict of key: agent_id, value: action
        """
        actions = {}
        for agent_controller in self.__agent_controllers:
            agent = agent_controller.agent
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(states[agent.id]), 0)
            action = agent_controller.action(tf_prev_state)
            actions.update({agent.id: list(action)})
        return actions

    def __stop_policy_controllers(self):
        for policy_controller in self.__policy_controllers:
            policy_controller.stop()

    def __is_done(self):
        return any(
            [
                ac.done(
                    [
                        agent_controller.agent
                        for agent_controller in self.__agent_controllers
                        if agent_controller.agent.agent_type != ac.agent.agent_type
                    ]
                )
                for ac in self.__agent_controllers
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
            for agent_controller in self.__agent_controllers
        }

    def __record_to_buffer(self, tuple: Tuple):
        """
        Records inside the replay_buffer given as parameter the observation tuple of the agents,
        where each agent is of a given type.
        :param tuple: tuple of (prev_states, actions, rewards, next_states)
        :return:
        """

        prev_states, actions, rewards, next_states = tuple
        prev_states_t, actions_t, rewards_t, next_states_t = [], [], [], []
        for agent_controller in self.__agent_controllers:
            agent = agent_controller.agent
            prev_states_t.append(prev_states[agent.id])
            actions_t.append(actions[agent.id])
            rewards_t.append(rewards[agent.id])
            next_states_t.append(next_states[agent.id])

        # logging.info(f"{agent_type} rewards: {rewards_t}")
        self.__buffer_controller.record(
            record_tuple=(prev_states_t, actions_t, rewards_t, next_states_t),
        )

    def __agent_by_id(self, agent_id: str):
        """
        Get the agent with the specified id.
        :param agent_id: agent id to search for
        :return: agent if any, None otherwise
        """
        return next(
            (agent for agent in self.__environment.agents if agent.id == agent_id), None
        )

    def __step_agent(self, agent_id: str, action: Tuple[float, float]):
        """
        Step an agent inside the environment given its action
        :param agent_id: id of the agent
        :param action: respective action
        :return:
        """
        agent = self.__agent_by_id(agent_id)
        v, turn = action[0], action[1]

        agent.vx, agent.vy = np.abs(v) * np.cos(turn), np.abs(v) * np.sin(turn)

        next_x, next_y = (
            agent.x + agent.vx * self.__t_step,
            agent.y + agent.vy * self.__t_step,
        )

        agent.x, agent.y = (
            np.clip(next_x, 0, self.__environment.x_dim - 1),
            np.clip(next_y, 0, self.__environment.y_dim - 1),
        )
