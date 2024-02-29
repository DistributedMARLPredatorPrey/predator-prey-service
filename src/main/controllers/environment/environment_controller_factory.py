from src.main.controllers.agents.predator_prey.predator import (
    PredatorControllerFactory,
)
from src.main.controllers.agents.predator_prey.prey import (
    PreyControllerFactory,
)
from src.main.controllers.buffer.buffer_controller import BufferController
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.model.environment.environment import Environment
from src.main.model.environment.params.environment_params import (
    EnvironmentParams,
    EnvironmentParamsFactory,
)


class EnvironmentControllerFactory:
    def __init__(self):
        self._default_env_params: EnvironmentParams = EnvironmentParamsFactory.standard_parameters()

    def create_predator_prey(self, env_params: EnvironmentParams = None) -> EnvironmentController:
        """
        Creates a random EnvironmentController, where the position of each agent inside the Environment is random.
        :param env_params: Environment parameters
        :return: random EnvironmentController
        """
        params = env_params if env_params is not None else self._default_env_params
        # Controllers
        predator_controllers = PredatorControllerFactory.create_from_params(params)
        prey_controllers = PreyControllerFactory.create_from_params(params)
        buffer_controller = BufferController()
        # Model
        environment = Environment(x_dim=params.x_dim, y_dim=params.y_dim,
                                  agents=[
                                      agent_controller.agent
                                      for agent_controller in predator_controllers + prey_controllers
                                  ])
        return EnvironmentController(
            environment=environment,
            agent_controllers=predator_controllers + prey_controllers,
            buffer_controller=buffer_controller
        )
