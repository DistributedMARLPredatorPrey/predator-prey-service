import os

from src.main.controllers.agents.policy.predator_prey.predator_prey_policy_controller_simulation import (
    PredatorPreyPolicyControllerSimulation,
)
from src.main.controllers.agents.policy.predator_prey.predator_prey_policy_controller_learning import (
    PredatorPreyPolicyControllerLearning,
)
from src.main.model.config.config_utils import PredatorPreyConfig
from src.main.controllers.agents.policy.agent_policy_controller import (
    AgentPolicyController,
)


class AgentPolicyControllerFactory:
    def __init__(self, project_root_path: str):
        self.__prey_actor_model_path: str = os.path.join(
            project_root_path, "src", "main", "resources", "prey.keras"
        )
        self.__predator_actor_model_path: str = os.path.join(
            project_root_path, "src", "main", "resources", "predator.keras"
        )

    def prey_policy_controller_learning(self, init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyControllerLearning(
            init=init,
            broker_host=PredatorPreyConfig()
            .learner_service_configuration()
            .pubsub_broker,
            actor_model_path=self.__prey_actor_model_path,
            routing_key="prey-actor-model",
        )

    def predator_policy_controller_learning(self, init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyControllerLearning(
            init=init,
            broker_host=PredatorPreyConfig()
            .learner_service_configuration()
            .pubsub_broker,
            actor_model_path=self.__predator_actor_model_path,
            routing_key="predator-actor-model",
        )

    def prey_policy_controller_simulation(self) -> AgentPolicyController:
        return PredatorPreyPolicyControllerSimulation(
            actor_model_path=self.__prey_actor_model_path
        )

    def predator_policy_controller_simulation(self) -> AgentPolicyController:
        return PredatorPreyPolicyControllerSimulation(
            actor_model_path=self.__predator_actor_model_path
        )
