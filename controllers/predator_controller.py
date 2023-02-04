from random import randint

import numpy as np

from controllers.buffer import Buffer
from model.environment import Environment
from model.predator import Predator
import tensorflow as tf
from tensorflow.keras import layers

from view import tracks


class PredatorController:
    rnd_state = 42

    def __init__(self, env=None, predator=None):
        self.env = env if env is not None else Environment()
        self.predator = predator if predator is not None \
            else Predator(x=randint(0, self.env.x_dim), y=randint(0, self.env.y_dim))

        # creating models
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        # we create the target model for double learning (to prevent a moving target phenomenon)
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
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

    def iterate(self, i):
        done = False
        prev_state = self.env.racer.reset()
        episodic_reward = 0
        self.mean_speed += prev_state[self.env.num_states - 1]
        while not done:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # our policy is always noisy
            action = self.policy(tf_prev_state)[0]
            # Get state and reward from the environment
            state, reward, done = self.env.step(action)

            # we distinguish between termination with failure (state = None) and succesfull termination on track completion
            # succesfull termination is stored as a normal tuple
            fail = done and len(state) < self.env.num_states
            self.buffer.record((prev_state, action, reward, fail, state))

            if not (done):
                self.mean_speed += state[self.env.num_states - 1]
            episodic_reward += reward

            if self.buffer.buffer_counter > self.batch_size:
                states, actions, rewards, dones, newstates = self.buffer.sample_batch()
                targetQ = rewards + (1 - dones) * self.gamma * \
                          (self.target_critic([newstates, self.target_actor(newstates)]))
                loss1 = self.critic_model.train_on_batch([states, actions], targetQ)
                loss2 = self.aux_model.train_on_batch(states)

                self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
            prev_state = state

        self.ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(self.ep_reward_list[-40:])
        print("Episode {}: Iterations {}, Avg. Reward = {}, Last reward = {}. Avg. speed = {}"
              .format(self.ep, i, avg_reward, episodic_reward, self.mean_speed / i))
        print("\n")

        if self.ep > 0 and self.ep % 40 == 0:
            print("## Evaluating policy ##")
            tracks.metrics_run(self.actor_model, 10)
        self.ep += 1

    def save(self):
        self.critic_model.save(self.weights_file_critic)
        self.actor_model.save(self.weights_file_actor)
