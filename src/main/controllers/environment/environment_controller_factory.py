from src.main.model.config.config_utils import ConfigUtils
from src.main.controllers.policy.agent_policy_controller_factory import (
    AgentPolicyControllerFactory,
)
from src.main.controllers.agents.predator_prey.predator.predator_controller_factory import (
    PredatorControllerFactory,
)
from src.main.controllers.agents.predator_prey.prey.prey_controller_factory import (
    PreyControllerFactory,
)
from src.main.controllers.replay_buffer.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.model.environment.environment import Environment


class EnvironmentControllerFactory:
    def __init__(self):
        config_utils = ConfigUtils()
        self._env_config = config_utils.environment_configuration()
        self._replay_buffer_config = config_utils.replay_buffer_configuration()

    def create_predator_prey(self) -> EnvironmentController:
        """
        Creates a random EnvironmentController, where the position of each agent
        inside the Environment is random.
        :return: random EnvironmentController
        """
        # Controllers
        ## Actor receivers
        policy_controller_factory = AgentPolicyControllerFactory()
        pred_actor_receiver_controller = (
            policy_controller_factory.predator_policy_controller()
        )
        prey_actor_receiver_controller = (
            policy_controller_factory.prey_policy_controller()
        )
        ## Predators and Preys
        predator_controllers = PredatorControllerFactory.create_from_params(
            self._env_config, pred_actor_receiver_controller
        )
        prey_controllers = PreyControllerFactory.create_from_params(
            self._env_config, prey_actor_receiver_controller
        )
        ## Buffer
        buffer_controller = ReplayBufferController(
            self._replay_buffer_config.replay_buffer_host,
            self._replay_buffer_config.replay_buffer_port,
        )

        # Model
        environment = Environment(
            x_dim=self._env_config.x_dim,
            y_dim=self._env_config.y_dim,
            agents=[
                agent_controller.agent
                for agent_controller in predator_controllers + prey_controllers
            ],
        )
        return EnvironmentController(
            environment=environment,
            agent_controllers=predator_controllers + prey_controllers,
            buffer_controller=buffer_controller,
        )
