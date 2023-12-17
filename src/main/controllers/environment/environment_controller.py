from typing import List, Dict, Any

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
        self.total_iterations = 50_000

    def _initial_states(self):
        init_states = {}
        for agent_controller in self.agent_controllers:
            init_states.update(
                {agent_controller.agent.id: agent_controller.state(self.environment.agents)}
            )
        return init_states

    def _actions(self, states):
        # Get the actions from the agents
        actions = {}
        for agent_controller in self.agent_controllers:
            agent = agent_controller.agent
            tf_prev_state = tf.expand_dims(
                tf.convert_to_tensor(states[agent.id].state), 0
            )
            action = agent_controller.policy(tf_prev_state)
            actions.update({agent.id: list(action)})
        return actions

    def _record_by_type(self, agent_type: AgentType,
                        buffer: Buffer,
                        prev_states, actions, rewards, next_states
                        ):
        prev_states_t, actions_t, rewards_t, next_states_t = [], [], [], []
        agents = [agent_controller.agent for agent_controller in self.agent_controllers if
                  agent_controller.agent.agent_type == agent_type]
        # avg_rewards = {}
        for agent in agents:
            prev_states_t += prev_states[agent.id].state
            actions_t += actions[agent.id]
            rewards_t.append(rewards[agent.id])
            next_states_t += next_states[agent.id].state
            # avg_rewards.update({agent.id: avg_rewards[agent.id] + rewards_t[agent.id]})
        # print(avg_rewards)
        buffer.record((prev_states_t, actions_t, rewards_t, next_states_t))

    def train(self):
        """
        Starts the training
        :return:
        """
        prev_states = self._initial_states()
        # Train
        for it in range(self.total_iterations):

            # avg_rewards = {agent.id: 0 for agent in self.environment.agents}

            for k in range(5):
                # Collect all agents action
                actions = self._actions(prev_states)
                # Move all the agents at once and get their rewards only after
                next_states = self._step(actions)
                rewards = self._rewards()
                print(rewards)
                # print([reward for reward in rewards])
                for i, agent_type in enumerate(AgentType):
                    self._record_by_type(agent_type, self.buffers[i], prev_states, actions, rewards, next_states)

            # print([(p_id, r / 10) for p_id, r in avg_rewards.items()])
            for learner in self.learners:
                learner.update()

    def _step(self, actions: Dict[str, List[float]]) -> Dict[str, State]:
        for (agent_id, action) in actions.items():
            self._step_agent(agent_id, action)
        states = {}
        for agent_controller in self.agent_controllers:
            states.update({agent_controller.agent.id: agent_controller.state(self.environment.agents)})
        return states

    def _rewards(self) -> dict[str, float]:
        return {
            agent_controller.agent.id: agent_controller.reward()
            for agent_controller in self.agent_controllers
        }

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
