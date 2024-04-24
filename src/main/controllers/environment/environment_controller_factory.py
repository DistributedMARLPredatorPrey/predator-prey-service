import os
from multiprocessing import Process, Pool
from threading import Thread

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
        self._policy_controller_factory = AgentPolicyControllerFactory()
        self._pred_actor_receiver_controller = None
        self._prey_actor_receiver_controller = None

    def _set_pred_actor_rec_controller(self):
        self._pred_actor_receiver_controller = (
            self._policy_controller_factory.predator_policy_controller()
        )

    def _set_prey_actor_rec_controller(self):
        self._prey_actor_receiver_controller = (
            self._policy_controller_factory.prey_policy_controller()
        )

    @staticmethod
    def f(v):
        policy_controller_factory = AgentPolicyControllerFactory()
        if v:
            policy_controller_factory.predator_policy_controller()
        else:
            policy_controller_factory.prey_policy_controller()

    def create_predator_prey(self) -> EnvironmentController:
        """
        Creates a random EnvironmentController, where the position of each agent
        inside the Environment is random.
        :return: random EnvironmentController
        """
        # Controllers
        ## Actor receivers
        print("Create actor receivers")

        # t1, t2 = (Process(target=self._set_pred_actor_rec_controller),
        #           Process(target=self._set_prey_actor_rec_controller))
        # t1.start()
        # t2.start()
        #
        # t1.join()
        # t2.join()

        with Pool(2) as p:
            p.map(self.f, [True, False])

        print("done")

        self._set_pred_actor_rec_controller()
        self._set_prey_actor_rec_controller()

        ## Predators and Preys
        print("Create pred and preys")
        predator_controllers = PredatorControllerFactory.create_from_params(
            self._env_config, self._pred_actor_receiver_controller
        )
        prey_controllers = PreyControllerFactory.create_from_params(
            self._env_config, self._prey_actor_receiver_controller
        )
        ## Buffer
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
        )
