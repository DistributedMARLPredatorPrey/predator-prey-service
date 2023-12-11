from random import uniform, randint

from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.agents.predator import Predator
from src.main.model.environment.environment import Environment


class EnvironmentControllerFactory:

    @staticmethod
    def create_random(x_dim=None, y_dim=None) -> EnvironmentController:

        env_x_dim, env_y_dim = (500, 500) \
            if (x_dim is None or y_dim is None) else (x_dim, y_dim)

        n_predators, n_preys = randint(1, 5), randint(1, 5)
        agents = []
        for i in range(n_predators):
            agents.append(Predator(id="predator_${id}".format(id=i),
                                   x=uniform(0, env_x_dim), y=uniform(0, env_y_dim),
                                   vx=0.2, vy=0.2, acc=0))

        return EnvironmentController(Environment(x_dim=env_x_dim, y_dim=env_y_dim, agents=agents))

    @staticmethod
    def from_existing(environment: Environment) -> EnvironmentController:
        return EnvironmentController(environment)
