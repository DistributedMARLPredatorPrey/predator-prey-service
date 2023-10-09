from random import randint

import numpy as np

from src.controllers.buffer import Buffer
from src.controllers.environment_controller import Environment
from src.model.agents.predator import Predator
import tensorflow as tf
from tensorflow.keras import layers
from src.view.pencil_of_lines import *

class PredatorController:
    rnd_state = 42

    def __init__(self, env=None, predator=None, actor_model=None, critic_model=None):
        self.env = env if env is not None else Environment()
        self.predator = predator if predator is not None \
            else Predator(x=randint(0, self.env.x_dim), y=randint(0, self.env.y_dim))

        # creating models
        self.actor_model = self.get_actor() if actor_model is None else actor_model
        self.critic_model = self.get_critic() if critic_model is None else critic_model

        # we create the target model for double learning (to prevent a moving target phenomenon)
        self.target_actor = self.get_actor() if actor_model is None else actor_model
        self.target_critic = self.get_critic() if critic_model is None else critic_model
        # making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # make target models non trainable.
        self.target_actor.trainable = False
        self.target_critic.trainable = False

        # Compose the models in a single one.
        self.aux_model = self.compose(self.actor_model, self.target_critic)

        # Setup the problem
        # Discount factor
        self.gamma = 0.99

        # Buffer settings
        self.buffer_dim = 50000
        self.batch_size = 64
        # Buffer
        self.buffer = Buffer(self.buffer_dim, self.batch_size, self.env.num_states, self.env.num_actions)

        # Hyperparameters
        self.total_iterations = 50_000

        # Target network parameter update factor, for double DQN
        self.tau = 0.005

        # Learning rate for actor-critic models
        self.critic_lr = 0.001
        self.aux_lr = 0.001

        self.is_training = False

        self.load_weights = True
        self.save_weights = False

        self.weights_file_actor = './predatormodel/{agent_id}/actormodel'.format(agent_id=self.predator.id)
        self.weights_file_critic = './predatormodel/{agent_id}/criticmodel'.format(agent_id=self.predator.id)

        # Define the optimizer
        critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        aux_optimizer = tf.keras.optimizers.Adam(self.aux_lr)

        # Compile the models
        self.critic_model.compile(loss='mse', optimizer=critic_optimizer)
        self.aux_model.compile(optimizer=aux_optimizer)

        self.mean_speed = 0
        self.ep = 0
        self.avg_reward = 0

        # History of rewards per episode
        self.ep_reward_list = []
        # Average reward history of last few episodes
        self.avg_reward_list = []

    # Slowly updating target parameters according to the tau rate <<1
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def update_weights(self, target_weights, weights, tau):
        return target_weights * (1 - tau) + weights * tau

    # We compose actor and critic in a single model.
    # The actor is trained by maximizing the future expected reward, estimated
    # by the critic. The critic should be freezed while training the actor.
    # For simplicitly, we just use the target critic, that is not trainable.
    def compose(self, actor, critic):
        state_input = layers.Input(shape=self.env.num_states)
        a = actor(state_input)
        q = critic([state_input, a])

        m = tf.keras.Model(state_input, q)
        # the loss function of the compound model is just the opposite of the critic output
        m.add_loss(-q)
        return m

    # The actor choose the move, given the state
    def get_actor(self, train_acceleration=True, train_direction=True):
        # the actor has separate towers for action and speed
        # in this way we can train them separately
        inputs = layers.Input(shape=(self.env.num_states,))
        out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(inputs)
        out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
        # acceleration
        out1 = layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)

        out2 = layers.Dense(32, activation="relu", trainable=train_direction)(inputs)
        out2 = layers.Dense(32, activation="relu", trainable=train_direction)(out2)
        # angular acceleration
        out2 = layers.Dense(1, activation='tanh', trainable=train_direction)(out2)

        outputs = layers.concatenate([out1, out2])

        # outputs = outputs * upper_bound #resize the range, if required
        model = tf.keras.Model(inputs, outputs, name="actor")
        return model

    # The critic compute the q-value, given the state and the action
    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.env.num_states)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=self.env.num_actions)
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1)(out)  # Outputs single value

        model = tf.keras.Model([state_input, action_input], outputs, name="critic")

        return model

    def policy(self, state, verbose=False):
        # the policy used for training just add noise to the action
        # the amount of noise is kept constant during training
        sampled_action = tf.squeeze(self.actor_model(state))
        noise = np.random.normal(scale=0.1, size=2)

        # we may change the amount of noise for actions during training
        noise[0] *= 2
        noise[1] *= .5

        # Adding noise to action
        sampled_action = sampled_action.numpy()
        sampled_action += noise

        # in verbose mode, we may print information about selected actions
        if verbose and sampled_action[0] < 0:
            print("decelerating")

        # Finally, we ensure actions are within bounds
        legal_action = np.clip(sampled_action, self.env.lower_bound, self.env.upper_bound)

        return [np.squeeze(legal_action)]

    def save(self):
        self.critic_model.save(self.weights_file_critic)
        self.actor_model.save(self.weights_file_actor)