import random

from src.main.controllers.agents.predator_controller import PredatorController
from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.environment import Environment
from src.main.model.agents.predator import Predator
from datetime import datetime


def train():
    x_dim, y_dim = 500, 500
    predators = [Predator("predator-%d".format(i),
                          random.randint(0, x_dim),
                          random.randint(0, y_dim)
                          )
                 for i in range(5)]
    env_controller: EnvironmentController = (
        EnvironmentController(Environment(x_dim, y_dim, predators)))
    predator_controllers = [PredatorController(env_controller, predator) for predator in predators]
    # train
    total_iterations = 50_000
    for it in range(total_iterations):
        for predator_controller in predator_controllers:
            predator_controller.iterate()


if __name__ == '__main__':
    start_t = datetime.now()
    train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t - start_t))
