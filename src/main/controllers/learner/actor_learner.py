from src.main.controllers.agents.buffer import Buffer
from src.main.controllers.learner.critic_learner import CriticLearner
from src.main.model.agents.neural_networks.critic import Critic
from src.main.model.agents.neural_networks.actor import Actor

import tensorflow as tf


class ActorLearner:

    def __init__(self, buffer: Buffer, num_states: int, num_actions: int):
        self.buffer = buffer
        self.actor_model = Actor(num_states).model
        self.critic_model = Critic(num_states, num_actions).model
        self.critic_learner = CriticLearner(self.buffer, self.actor_model.get_weights(), num_states, num_actions)
        self.actor_lr = 5e-5
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def learn(self):
        critic_weights, state_batch = self.critic_learner.learn()
        self.critic_model.set_weights(critic_weights)
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
