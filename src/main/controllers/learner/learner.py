from typing import List

import tensorflow as tf
import numpy as np
from tensorflow import GradientTape

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.neural_networks.actor import Actor
from src.main.model.agents.neural_networks.critic import Critic
from src.main.controllers.agents.buffer import Buffer


class Learner:

    def __init__(self, buffer: Buffer, par_services: List[ParameterService], num_states: int, num_actions: int,
                 num_agents: int):
        # Parameters
        self.buffer = buffer
        self.par_services = par_services
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = num_agents

        # Learning rate for actor-critic models
        self.critic_lr = 1e-4
        self.actor_lr = 5e-5

        # Creating Optimizer for Actor and Critic networks
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Creating critic models
        self.critic_models, self.target_critics = [], []
        for j in range(num_agents):
            self.critic_models.append(Critic(j, num_states, num_actions, num_agents).model)
            self.target_critics.append(Critic(j, num_states, num_actions, num_agents).model)
            self.target_critics[j].set_weights(self.critic_models[j].get_weights())

            self.target_critics[j].trainable = False
            self.critic_models[j].compile(loss='mse', optimizer=self.critic_optimizer)

        # Compile the models
        # self.critic_model.compile(loss='mse', optimizer=self.critic_optimizer)

        # creating target actor model
        self.actor_models, self.target_actors = [], []
        for j in range(num_agents):
            self.actor_models.append(Actor(j, num_states).model)
            self.target_actors.append(Actor(j, num_states).model)
            self.target_actors[j].set_weights(self.actor_models[j].get_weights())

            self.par_services[j].set_model(self.actor_models[j])

            # Make target models non trainable
            self.target_actors[j].trainable = False
            self.actor_models[j].compile(loss='mse', optimizer=self.actor_optimizer)

        # Discount factor for future rewards
        self.gamma = 0.95

        # Used to update target networks
        self.tau = 0.005

    def update(self):
        self._update_actors(self._update_critic())
        for j in range(self.num_agents):
            self._update_target(self.target_actors[j].variables, self.actor_models[j].variables, self.tau)
            self._update_target(self.target_critics[j].variables, self.critic_models[j].variables, self.tau)

    # Slowly updating target parameters according to the tau rate <<1
    @tf.function
    def _update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    # We compute the loss and update parameters of the critic network
    # It returns the updated critic network and the state batch, to be used by the actor
    def _update_critic(self):
        # Batch a sample from the buffer
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample_batch()

        # (:, [s_0_0, ..., s_0_13], [s_1_0, ..., s_1_13], ..., [s_9_0, ..., s_9_13])
        # print(state_batch.shape)
        # (:, [a_0_0, a_0_1], ..., [a_9_0, a_9_1])
        # print(action_batch.shape)
        # print(reward_batch.shape)
        # print(next_state_batch.shape)

        target_actions = np.zeros([self.buffer.batch_size, self.num_actions * self.num_agents])

        for j in range(self.num_agents):
            target_actions[:, j * self.num_actions: (j + 1) * self.num_actions] = self.target_actors[j](
                # get the next state of the j-agent
                next_state_batch[:, j * self.num_states: (j + 1) * self.num_states], training=True
            )

        for i in range(self.num_agents):
            # Train the Critic network
            with tf.GradientTape() as tape:
                y = reward_batch[:, i] + self.gamma * self.target_critics[i](
                    [next_state_batch, target_actions], training=True
                )
                critic_value = self.critic_models[i]([state_batch, action_batch], training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, self.critic_models[i].trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_models[i].trainable_variables)
            )
            # Return the updated Critic along with the used state batch used to apply a gradient update
            return state_batch

    def _update_actors(self, state_batch):

        # with tf.GradientTape() as tape:
        # actions = tf.zeros((self.buffer.batch_size, self.num_agents * self.num_actions))
        # for j in range(self.num_agents):
        #
        #     start_col = j * self.num_states
        #     end_col = (j + 1) * self.num_states
        #
        #     actor_output = self.actor_models[j](state_batch[:, start_col: end_col])
        #
        #     batch_indices = tf.range(self.buffer.batch_size)
        #     col_indices = tf.range(start_col, end_col)
        #
        #     indices_to_update = tf.stack([batch_indices, col_indices], axis=1)
        #     actions = tf.tensor_scatter_nd_update(actions, indices_to_update, actor_output)

        # indices_to_update = tf.meshgrid(
        #    tf.range(0, self.buffer.batch_size),
        #    tf.range(j * self.num_actions, (j + 1) * self.num_actions)
        # )
        # indices_to_update = tf.stack([indices_to_update[0], indices_to_update[1]], axis=-1)
        # print(indices_to_update)#

        # actions = tf.tensor_scatter_nd_update(actions, indices_to_update, [self.actor_models[j](
        #    state_batch[:, j * self.num_states: (j + 1) * self.num_states]
        # )])
        # action_tensor = tf.convert_to_tensor(actions)

        actions = np.zeros((self.buffer.batch_size, self.num_agents * self.num_actions))
        for j in range(self.num_agents):
            single_action = self.actor_models[j](
                state_batch[:, j * self.num_states: (j + 1) * self.num_states],
                training=True
            )
            actions[:, j * self.num_actions: (j + 1) * self.num_actions] = single_action

            for i in range(self.num_agents):

                with GradientTape() as tape:

                    joint_actions = np.array([
                        np.array([joint_action[k: k + self.num_actions]
                                  if k != i * self.num_actions
                                  else
                                  self.actor_models[i](
                                      np.array([state_batch[idx][i * self.num_states: (i + 1) * self.num_states]]),
                                      training=True
                                  )[0]
                                  for k in range(0, len(joint_action), self.num_actions)]).flatten()
                        for idx, joint_action in enumerate(actions)
                    ])

                    critic_values = self.critic_models[i]([state_batch, joint_actions], training=True)
                    actor_loss = -tf.math.reduce_mean(critic_values)

                # print([var.name for var in tape.watched_variables()])
                actor_grad = tape.gradient(actor_loss, self.actor_models[i].trainable_variables)
                print(actor_grad)
                self.actor_optimizer.apply_gradients(
                    zip(actor_grad, self.actor_models[i].trainable_variables)
                )

