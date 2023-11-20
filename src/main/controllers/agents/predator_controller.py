import numpy as np
import tensorflow as tf

from main.controllers.learner.learner import Learner
from main.controllers.parameter_server.parameter_service import ParameterService
from src.main.controllers.agents.buffer import Buffer
from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.agents.neural_networks.actor import Actor
from src.main.model.agents.neural_networks.critic import Critic
from src.main.model.agents.predator import Predator


class PredatorController:
    rnd_state = 42

    def __init__(self, env_controller: EnvironmentController, predator: Predator):
        self.env_controller = env_controller
        self.predator = predator

        # initial state
        self.prev_state, _, _ = env_controller.observe(self.predator)
        self.episodic_reward = 0

        # Buffer
        self.buffer = Buffer(50_000, 64,
                             self.env_controller.environment.num_states,
                             self.env_controller.environment.num_actions)
        # ParameterService & Learner
        self.par_service = ParameterService()
        self.learner = Learner(self.buffer, self.par_service,
                               self.env_controller.environment.num_states,
                               self.env_controller.environment.num_actions)

        # Hyperparameters
        self.total_iterations = 50_000

        self.mean_speed = 0
        self.ep = 0
        self.avg_reward = 0

        # History of rewards per episode
        self.ep_reward_list = []
        # Average reward history of last few episodes
        self.avg_reward_list = []

    def _policy(self, state, verbose=False):
        # the policy used for training just add noise to the action
        # the amount of noise is kept constant during training
        sampled_action = tf.squeeze(self.par_service.actor_model(state))
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

        return np.squeeze(legal_action)

    def next_action(self):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_state), 0)
        action = self._policy(tf_prev_state)

        # Receive state and reward from environment
        state, done, reward = self.env_controller.step(self.predator, action)
        print("pos: ({}, {}) a: {}, state: {}, reward: {}"
              .format(self.predator.x, self.predator.y, action, state, reward))

        self.buffer.record((self.prev_state, action, reward, state))
        self.episodic_reward += reward
        # Update
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample_batch()
        self.update(state_batch, action_batch, reward_batch, next_state_batch)

        self.prev_state = state
