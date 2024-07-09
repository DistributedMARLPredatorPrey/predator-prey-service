from src.main.controllers.agents.predator_prey.predator.predator_controller_factory import (
    PredatorControllerFactory,
)
from src.main.controllers.agents.predator_prey.prey.prey_controller_factory import (
    PreyControllerFactory,
)
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from main.controllers.environment.utils.environment_controller_utils import (
    EnvironmentControllerUtils,
)
from src.main.controllers.environment.utils.predator_prey_utils.predator_prey_utils import (
    PredatorPreyUtils,
)
from src.main.controllers.policy.agent_policy_controller_factory import (
    AgentPolicyControllerFactory,
)
from src.main.controllers.replay_buffer.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.model.config.config_utils import ConfigUtils
from src.main.model.environment.environment import Environment


class EnvironmentControllerFactory:
    def __init__(self):
        config_utils = ConfigUtils()
        self._env_config = config_utils.environment_configuration()
        self._replay_buffer_config = config_utils.replay_buffer_configuration()
        self._policy_controller_factory = AgentPolicyControllerFactory()
        self._pred_actor_receiver_controller = None
        self._prey_actor_receiver_controller = None

    def create_predator_prey(self) -> EnvironmentController:
        """
        Creates a random EnvironmentController, where the position of each agent
        inside the Environment is random.
        :return: random EnvironmentController
        """
        # Controllers
        print("Create actor receivers")
        utils = PredatorPreyUtils()
        utils.initialize_policy_receivers()
        factory = AgentPolicyControllerFactory()
        self._prey_actor_receiver_controller, self._pred_actor_receiver_controller = (
            factory.prey_policy_controller(),
            factory.predator_policy_controller(),
        )

        # Predators and Preys
        print("Create pred and preys")
        predator_controllers = PredatorControllerFactory.create_from_params(
            self._env_config, self._pred_actor_receiver_controller
        )
        prey_controllers = PreyControllerFactory.create_from_params(
            self._env_config, self._prey_actor_receiver_controller
        )
        # Buffer
        print("Create buffer contr")
        buffer_controller = ReplayBufferController(
            self._replay_buffer_config.replay_buffer_host,
            self._replay_buffer_config.replay_buffer_port,
        )
        # Model
        print("Create env")
        environment = Environment(
            x_dim=self._env_config.x_dim,
            y_dim=self._env_config.y_dim,
            agents=[
                agent_controller.agent
                for agent_controller in predator_controllers + prey_controllers
            ],
        )
        print("Create env controller")
        return EnvironmentController(
            environment=environment,
            agent_controllers=predator_controllers + prey_controllers,
            buffer_controller=buffer_controller,
            env_controller_utils=EnvironmentControllerUtils(
                self._env_config.base_experiment_path,
                self._env_config.rel_experiment_path,
            ),
        )
