from src.main.controllers.agents.predator.predator_controller_factory import (
    PredatorControllerFactory,
)
from src.main.controllers.agents.prey.prey_controller_factory import (
    PreyControllerFactory,
)
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.controllers.learner.learner_factory import LearnerFactory
from src.main.model.environment.buffer.buffer_factory import BufferFactory
from src.main.model.environment.environment import Environment
from src.main.model.environment.params.environment_params import (
    EnvironmentParams,
    EnvironmentParamsFactory,
)


class EnvironmentControllerFactory:
    def __init__(self):
        self._default_env_params: EnvironmentParams = (
            EnvironmentParamsFactory.standard_parameters()
        )

    def create_random(
        self, env_params: EnvironmentParams = None
    ) -> EnvironmentController:
        """
        Creates a random EnvironmentController, where the position of each agent inside the Environment is random.
        :param env_params: Environment parameters
        :return: random EnvironmentController
        """
        params = env_params if env_params is not None else self._default_env_params
        predator_controllers = PredatorControllerFactory.create_from_params(params)
        prey_controllers = PreyControllerFactory.create_from_params(params)
        environment = Environment(
            x_dim=params.x_dim,
            y_dim=params.y_dim,
            agents=[
                agent_controller.agent
                for agent_controller in predator_controllers + prey_controllers
            ],
        )
        # Create two buffers, one for the predators and the other for the preys
        buffers = BufferFactory.create_buffers(
            num_states=params.num_states,
            num_actions=params.num_actions,
            sizes=[len(predator_controllers), len(prey_controllers)],
        )
        # Create two learners, one for the predators and the other for the preys
        learners = LearnerFactory.create_learners(
            buffers,
            [
                [
                    predator_controller.par_service
                    for predator_controller in predator_controllers
                ],
                [prey_controller.par_service for prey_controller in prey_controllers],
            ],
            params.num_states,
            params.num_actions,
        )
        return EnvironmentController(
            environment=environment,
            agent_controllers=predator_controllers + prey_controllers,
            buffers=buffers,
            learners=learners,
        )

    @staticmethod
    def from_existing(environment: Environment) -> EnvironmentController:
        """
        TODO: it should create an EnvironmentController based on previous computation, i.e. where the neural networks
        are partially trained
        :param environment: existing Environment
        :return:
        """
        return EnvironmentController(environment)
