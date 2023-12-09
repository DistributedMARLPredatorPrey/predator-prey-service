import random

import numpy as np
import tensorflow as tf

from src.main.controllers.agents.buffer import Buffer
from src.main.controllers.learner.learner import Learner
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.controllers.agents.predator_controller import PredatorController
from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.environment.environment import Environment
from src.main.model.agents.predator import Predator
from datetime import datetime


def train():
    x_dim, y_dim = 250, 250
    predators = [Predator(i, random.randint(0, x_dim), random.randint(0, y_dim)) for i in range(10)]

    # ParameterService & Learner
    par_services = [ParameterService() for _ in range(len(predators))]

    env_controller: EnvironmentController = (
        EnvironmentController(Environment(x_dim, y_dim, predators)))
    predator_controllers = [PredatorController(env_controller, predator, par_services[predators.index(predator)])
                            for predator in predators]

    # Buffer
    buffer = Buffer(50_000, 64,
                    env_controller.environment.num_states,
                    env_controller.environment.num_actions,
                    len(predators))

    learners = [Learner(buffer,
                        par_services,
                        env_controller.environment.num_states,
                        env_controller.environment.num_actions,
                        len(predators)
                        )
                ]

    # Initial observation
    prev_obs_dict = {}
    for agent in env_controller.environment.agents:
        prev_obs_dict.update({agent.id: env_controller.observe(agent)})

    # Train
    total_iterations = 50_000
    for it in range(total_iterations):

        for k in range(50):
            # Get the actions from the agents
            actions_dict = {}
            for predator_controller in predator_controllers:
                p_id = predator_controller.predator.id
                tf_prev_state = tf.expand_dims(
                    tf.convert_to_tensor(prev_obs_dict[p_id].observation), 0
                )
                action = predator_controller.policy(tf_prev_state)
                actions_dict.update({p_id: list(action)})

            # Move all the agents at once and get their rewards only after
            next_obs_dict = env_controller.step(actions_dict)
            rewards_dict = env_controller.rewards()

            prev_obs, actions, rewards, next_obs = [], [], [], []
            for agent in env_controller.environment.agents:
                p_id = agent.id
                prev_obs += prev_obs_dict[p_id].observation
                actions += actions_dict[p_id]
                rewards.append(rewards_dict[p_id])
                next_obs += next_obs_dict[p_id].observation

            # Store on the buffer the joint data
            buffer.record((prev_obs, actions, rewards, next_obs))
            for learner in learners:
                learner.update()


# for predator_controller in predator_controllers:
#    predator_controller.ep_reward_list.append(predator_controller.episodic_reward)
#    avg_reward = np.mean(predator_controller.episodic_reward)
#    print("iteration {}, id: {}, avg reward: {}"
#          .format(it, predator_controller.predator.id, avg_reward))
#    predator_controller.episodic_reward = 0


if __name__ == '__main__':
    start_t = datetime.now()
    train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t - start_t))
