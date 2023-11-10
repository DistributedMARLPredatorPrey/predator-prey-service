import numpy as np
import tensorflow as tf
from keras import Model

from main.controllers.agents.buffer import Buffer
from main.controllers.environment.environment_controller import EnvironmentController
from main.model.agents.predator import Predator


class PredatorController:
    rnd_state = 42

    def __init__(self, env_controller: EnvironmentController, predator: Predator,
                 actor_model: Model = None, critic_model: Model = None):
        self.env_controller = env_controller
        self.predator = predator

        # initial state
        self.prev_state = env_controller.observe(self.predator)
        self.episodic_reward = 0

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
        legal_action = np.clip(sampled_action,
                               self.env_controller.lower_bound,
                               self.env_controller.upper_bound)

        return [np.squeeze(legal_action)]

    def save(self):
        self.critic_model.save(self.weights_file_critic)
        self.actor_model.save(self.weights_file_actor)

    def iterate(self):

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_state), 0)
        action = self.policy(tf_prev_state)

        # Receive state and reward from environment
        state, reward, done, info = self.env_controller.step(self.predator, action)

        self.buffer.record((self.prev_state, action, reward, state))
        self.episodic_reward += reward

        self.buffer.learn()
        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state
