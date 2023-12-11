import numpy as np
import tensorflow as tf

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator


class PredatorController:

    def __init__(self, lower_bound: float, upper_bound: float, predator: Predator,
                 par_service: ParameterService):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.predator = predator

        # Initial state
        self.episodic_reward = 0
        self.par_service = par_service

        # Hyperparameters
        self.total_iterations = 50_000

        self.mean_speed = 0
        self.ep = 0
        self.avg_reward = 0

        # History of rewards per episode
        self.ep_reward_list = []
        # Average reward history of last few episodes
        self.avg_reward_list = []

    def policy(self, state, verbose=False):
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
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)

        return np.squeeze(legal_action)
