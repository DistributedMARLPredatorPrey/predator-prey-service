from src.main.model.config.config import EnvironmentConfig
from src.main.controllers.replay_buffer.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.controllers.agents.policy.agent_policy_controller import (
    AgentPolicyController,
)
from src.main.controllers.agents.predator_prey.agent_controller_factory import (
    AgentControllerFactory,
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
from src.main.controllers.agents.policy.agent_policy_controller_factory import (
    AgentPolicyControllerFactory,
)
from src.main.controllers.replay_buffer.remote.remote_replay_buffer_controller import (
    RemoteReplayBufferController,
)
from src.main.model.config.config_utils import PredatorPreyConfig
from src.main.model.environment.environment import Environment


class EnvironmentControllerFactory:
    def create_predator_prey_learning(
        self, pred_prey_config: PredatorPreyConfig, init: bool = True
    ) -> EnvironmentController:
        """
        Creates a random EnvironmentController in learning mode, where the position of each agent
        inside the Environment is random.
        :param pred_prey_config: PredatorPreyConfig
        :param init: Should be True if it's the first run
        :return: EnvironmentController
        """
        env_config, replay_buffer_config = (
            pred_prey_config.environment_configuration(),
            pred_prey_config.replay_buffer_configuration(),
        )
        if init:
            utils = PredatorPreyUtils()
            utils.initialize_policy_receivers()
        buffer_controller = RemoteReplayBufferController(
            replay_buffer_config.replay_buffer_host,
            replay_buffer_config.replay_buffer_port,
        )
        policy_controller_factory = AgentPolicyControllerFactory()
        return self.__create_predator_prey(
            env_config=env_config,
            prey_policy_controller=policy_controller_factory.prey_policy_controller_learning(
                init=False
            ),
            pred_policy_controller=policy_controller_factory.predator_policy_controller_learning(
                init=False
            ),
            buffer_controller=buffer_controller,
        )

    def create_predator_prey_simulation(
        self, pred_prey_config: PredatorPreyConfig
    ) -> EnvironmentController:
        """
        Creates a random EnvironmentController in simulation mode, where the position of each agent
        inside the Environment is random.
        :param pred_prey_config: PredatorPreyConfig
        :return: EnvironmentController
        """
        policy_controller_factory = AgentPolicyControllerFactory()
        return self.__create_predator_prey(
            env_config=pred_prey_config.environment_configuration(),
            prey_policy_controller=policy_controller_factory.prey_policy_controller_simulation(),
            pred_policy_controller=policy_controller_factory.predator_policy_controller_simulation(),
            buffer_controller=None,
        )

    def __create_predator_prey(
        self,
        env_config: EnvironmentConfig,
        prey_policy_controller: AgentPolicyController,
        pred_policy_controller: AgentPolicyController,
        buffer_controller: ReplayBufferController,
    ):
        predator_controllers = AgentControllerFactory.predator_controllers_from_config(
            env_config, pred_policy_controller
        )
        prey_controllers = AgentControllerFactory.prey_controllers_from_config(
            env_config, prey_policy_controller
        )
        environment = Environment(
            x_dim=env_config.x_dim,
            y_dim=env_config.y_dim,
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
                env_config.base_experiment_path,
                env_config.rel_experiment_path,
            ),
        )
