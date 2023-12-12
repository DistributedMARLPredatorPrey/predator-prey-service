from dataclasses import dataclass
from random import uniform, randint
from typing import Tuple, List

from src.main.controllers.learner.learner import Learner
from src.main.controllers.agents.agent_controller import AgentController
from src.main.model.agents.agent import Agent
from src.main.controllers.agents.PreyController import PreyController
from src.main.model.agents.prey import Prey
from src.main.controllers.agents.buffer import Buffer
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.controllers.agents.predator_controller import PredatorController
from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.model.agents.predator import Predator
from src.main.model.environment.environment import Environment


class EnvironmentControllerFactory:
    @dataclass(frozen=True)
    class Parameters:
        env_x_dim: int
        env_y_dim: int
        upper_bound: int
        lower_bound: int
        n_predators: int
        n_preys: int

    @staticmethod
    def create_random(x_dim=None, y_dim=None) -> EnvironmentController:

        env_x_dim, env_y_dim = (250, 250) \
            if (x_dim is None or y_dim is None) else (x_dim, y_dim)

        params = EnvironmentControllerFactory.Parameters(
            env_x_dim=env_x_dim, env_y_dim=env_y_dim, upper_bound=1, lower_bound=-1, n_predators=10, n_preys=10
        )

        # (List[Predator], List[PredatorController], List[ParameterService])
        predator_parameters = EnvironmentControllerFactory._get_predator_parameters(params)
        prey_parameters = EnvironmentControllerFactory._get_prey_parameters(params)
        environment = Environment(x_dim=env_x_dim, y_dim=env_y_dim,
                                  agents=predator_parameters[0] + prey_parameters[0])
        # Create two buffers, one for the predators and the other for the preys
        buffers = EnvironmentControllerFactory._get_buffers(
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            sizes=[len(predator_parameters[0]), len(prey_parameters[0])]
        )
        # Create two learners, one for the predators and the other for the preys
        learners = EnvironmentControllerFactory._get_learners(buffers,
                                                              [predator_parameters[2], prey_parameters[2]],
                                                              environment.num_states,
                                                              environment.num_actions
                                                              )
        return EnvironmentController(environment=environment,
                                     par_services=predator_parameters[2] + prey_parameters[2],
                                     agent_controllers=predator_parameters[1] + prey_parameters[1],
                                     buffers=buffers,
                                     learners=learners
                                     )

    @staticmethod
    def from_existing(environment: Environment) -> EnvironmentController:
        return EnvironmentController(environment)

    @staticmethod
    def _get_predator_parameters(params: Parameters) -> (
            Tuple)[List[Predator], List[PredatorController], List[ParameterService]]:
        predators, predator_controllers = [], []
        par_services = []
        for i in range(params.n_predators):
            predator = Predator(id=f"predator_{i}",
                                x=uniform(0, params.env_x_dim),
                                y=uniform(0, params.env_y_dim)
                                )
            par_service = ParameterService()
            predator_controllers.append(
                PredatorController(
                    lower_bound=params.lower_bound,
                    upper_bound=params.upper_bound,
                    r=10,
                    life=100,
                    predator=predator,
                    par_service=par_service
                )
            )
            predators.append(predator)
            par_services.append(par_service)
        return predators, predator_controllers, par_services

    @staticmethod
    def _get_prey_parameters(params: Parameters) -> (
            Tuple)[List[Prey], List[PreyController], List[ParameterService]]:
        preys, prey_controllers, par_services = [], [], []
        for i in range(params.n_predators):
            prey = Prey(id=f"prey_{i}", x=uniform(0, params.env_x_dim), y=uniform(0, params.env_y_dim))
            par_service = ParameterService()
            prey_controllers.append(
                PreyController(
                    lower_bound=params.lower_bound,
                    upper_bound=params.upper_bound,
                    r=10,
                    life=100,
                    prey=prey,
                    par_service=par_service
                )
            )
            preys.append(prey)
            par_services.append(par_service)
        return preys, prey_controllers, par_services

    @staticmethod
    def _get_buffers(num_states, num_actions, sizes):
        return [Buffer(50_000, 64, num_states, num_actions, size) for size in sizes]

    @staticmethod
    def _get_learners(buffers, par_services, num_states, num_actions):
        return [Learner(buffers[i], par_services[i], num_states, num_actions, len(par_services[i]))
                for i in range(len(buffers))]
