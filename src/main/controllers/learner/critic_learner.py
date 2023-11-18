import tensorflow as tf
import numpy as np

from main.model.agents.neural_networks.actor import Actor
from main.model.agents.neural_networks.critic import Critic
from src.main.controllers.agents.buffer import Buffer


class CriticLearner:

    def __init__(self, buffer: Buffer, actor_weights: list, num_states: int, num_actions: int):
        # Parameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.buffer = buffer

        # Creating critic models
        self.critic_model = Critic(num_states, num_actions).model
        self.target_critic = Critic(num_states, num_actions).model
        self.target_critic.set_weights(self.critic_model.get_weights())

        # creating target actor model
        self.target_actor = Actor(num_states).model
        self.target_actor.set_weights(actor_weights)

        # Make target models non trainable.
        self.target_actor.trainable = False
        self.target_critic.trainable = False

        # Learning rate for actor-critic models
        self.critic_lr = 1e-4
        self.actor_lr = 5e-5

        # Creating Optimizer for actor and critic networks
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Discount factor for future rewards
        self.gamma = 0.95

        # Used to update target networks
        self.tau = 0.005

    # We compute the loss and update parameters of the critic network
    # It returns the updated critic network and the state batch, to be used by the actor
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        # Training  and Updating ***critic model***
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
        # Updating and training of ***critic network*** ended
        return self.critic_model, state_batch





