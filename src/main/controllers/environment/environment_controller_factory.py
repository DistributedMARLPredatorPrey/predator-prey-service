from random import uniform, randint

from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.agents.predator import Predator
from src.main.model.environment.environment import Environment


class EnvironmentControllerFactory:

    @staticmethod
    def create_random(x_dim=None, y_dim=None) -> EnvironmentController:
        env_x_dim, env_y_dim = (250, 250) \
            if (x_dim is None or y_dim is None) else (x_dim, y_dim)

        n_predators, n_preys = 10, randint(1, 5)
        agents = []
        for i in range(n_predators):
            agents.append(Predator(id=f"predator_{i}",
                                   x=uniform(0, env_x_dim),
                                   y=uniform(0, env_y_dim)
                                   )
                          )
        return EnvironmentController(Environment(x_dim=env_x_dim, y_dim=env_y_dim, agents=agents))

    @staticmethod
    def from_existing(environment: Environment) -> EnvironmentController:
        return EnvironmentController(environment)
