import logging

from src.main.controllers.policy.agent_policy_controller import AgentPolicyController
from src.main.controllers.agents.predator_prey.predator.predator_controller_factory import (
    PredatorControllerFactory,
)
from src.main.controllers.agents.predator_prey.prey.prey_controller_factory import (
    PreyControllerFactory,
)
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.controllers.environment.utils.environment_controller_utils import (
    EnvironmentControllerUtils,
)
from src.main.controllers.environment.utils.predator_prey_utils.predator_prey_utils import (
    PredatorPreyUtils,
)
from src.main.controllers.policy.agent_policy_controller_factory import (
    AgentPolicyControllerFactory,
)
from src.main.controllers.replay_buffer.remote.remote_replay_buffer_controller import (
    RemoteReplayBufferController,
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

    def create_predator_prey_learning(self, init: bool = True) -> EnvironmentController:
        """
        Creates a random EnvironmentController in learning mode, where the position of each agent
        inside the Environment is random.
        :return: random EnvironmentController
        """
        # Controllers
        logging.info("Creating Environment")
        if init:
            utils = PredatorPreyUtils()
            utils.initialize_policy_receivers()
        policy_controller_factory = AgentPolicyControllerFactory()
        return self.__create_predator_prey(
            prey_policy_controller=policy_controller_factory.prey_policy_controller_learning(
                init=False
            ),
            pred_policy_controller=policy_controller_factory.predator_policy_controller_learning(
                init=False
            ),
        )

    def create_predator_prey_simulation(self):
        """
        Creates a random EnvironmentController in simulation mode, where the position of each agent
        inside the Environment is random.
        :return: random EnvironmentController
        """
        policy_controller_factory = AgentPolicyControllerFactory()
        return self.__create_predator_prey(
            prey_policy_controller=policy_controller_factory.prey_policy_controller_simulation(),
            pred_policy_controller=policy_controller_factory.predator_policy_controller_simulation(),
        )

    def __create_predator_prey(
        self,
        prey_policy_controller: AgentPolicyController,
        pred_policy_controller: AgentPolicyController,
    ):
        predator_controllers = PredatorControllerFactory.create_from_params(
            self._env_config, pred_policy_controller
        )
        prey_controllers = PreyControllerFactory.create_from_params(
            self._env_config, prey_policy_controller
        )
        buffer_controller = RemoteReplayBufferController(
            self._replay_buffer_config.replay_buffer_host,
            self._replay_buffer_config.replay_buffer_port,
        )
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
            policy_controllers=[prey_policy_controller, pred_policy_controller],
            env_controller_utils=EnvironmentControllerUtils(
                self._env_config.base_experiment_path,
                self._env_config.rel_experiment_path,
            ),
        )
