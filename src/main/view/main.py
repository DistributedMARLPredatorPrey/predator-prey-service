import random

import numpy as np

from src.main.controllers.agents.predator_controller import PredatorController
from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.environment import Environment
from src.main.model.agents.predator import Predator
from datetime import datetime


def train():
    x_dim, y_dim = 500, 500
    predators = [Predator("predator-{}".format(i),
                          random.randint(0, x_dim),
                          random.randint(0, y_dim)
                          )
                 for i in range(10)]

    env_controller: EnvironmentController = (
        EnvironmentController(Environment(x_dim, y_dim, predators)))
    predator_controllers = [PredatorController(env_controller, predator) for predator in predators]
    # train
    total_iterations = 50_000
    for it in range(total_iterations):
        for k in range(50):
            for predator_controller in predator_controllers:
                predator_controller.iterate()

        for predator_controller in predator_controllers:
            predator_controller.ep_reward_list.append(predator_controller.episodic_reward)
            avg_reward = np.mean(predator_controller.episodic_reward)
            print("iteration {}, id: {}, avg reward: {}"
                  .format(it, predator_controller.predator.id, avg_reward))
            predator_controller.episodic_reward = 0


if __name__ == '__main__':
    start_t = datetime.now()
    train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t - start_t))
