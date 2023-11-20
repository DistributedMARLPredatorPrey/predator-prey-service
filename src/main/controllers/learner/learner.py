import tensorflow as tf
import numpy as np

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.neural_networks.actor import Actor
from src.main.model.agents.neural_networks.critic import Critic
from src.main.controllers.agents.buffer import Buffer


class Learner:

    def __init__(self, buffer: Buffer, par_service: ParameterService, num_states: int, num_actions: int):
        # Parameters
        self.buffer = buffer
        self.par_service = par_service
        self.num_states = num_states
        self.num_actions = num_actions

        # Creating critic models
        self.critic_model = Critic(num_states, num_actions).model
        self.target_critic = Critic(num_states, num_actions).model
        self.target_critic.set_weights(self.critic_model.get_weights())

        # creating target actor model
        self.actor_model = Actor(num_states).model
        self.target_actor = Actor(num_states).model
        self.target_actor.set_weights(self.actor_model.get_weights())

        # Make target models non trainable.
        self.target_actor.trainable = False
        self.target_critic.trainable = False

        # Learning rate for actor-critic models
        self.critic_lr = 1e-4
        self.actor_lr = 5e-5

        # Creating Optimizer for Actor and Critic networks
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Compile the models
        self.critic_model.compile(loss='mse', optimizer=self.critic_optimizer)
        self.actor_model.compile(optimizer=self.actor_optimizer)

        # Discount factor for future rewards
        self.gamma = 0.95

        # Used to update target networks
        self.tau = 0.005

    def update(self):
        self._update_actor(self._update_critic())
        self._update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self._update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
        self.par_service.set_model(self.actor_model)

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
        # Train the Critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        # Return the updated Critic along with the used state batch used to apply a gradient update
        return state_batch

    def _update_actor(self, state_batch):
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

