from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.agent import Agent

import tensorflow as tf
import numpy as np


class AgentController:

    def __init__(self, lower_bound: float, upper_bound: float,
                 agent: Agent,
                 par_service: ParameterService):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.agent = agent
        self.par_service = par_service

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

    # Base reward method, to be overridden by subclasses
    def reward(self, observation):
        raise NotImplementedError("Subclasses must implement this method")
