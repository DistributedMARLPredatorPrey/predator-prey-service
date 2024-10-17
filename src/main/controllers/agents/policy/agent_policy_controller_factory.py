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

    __base_path: str = os.path.join("src", "main", "resources")

    def prey_policy_controller_learning(self, init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyControllerLearning(
            init=init,
            broker_host=PredatorPreyConfig()
            .learner_service_configuration()
            .pubsub_broker,
            actor_model_path=os.path.join(self.__base_path, f"prey_{os.environ.get('REL_PATH')}.keras"),
            routing_key="prey-actor-model",
        )

    def predator_policy_controller_learning(self, init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyControllerLearning(
            init=init,
            broker_host=PredatorPreyConfig()
            .learner_service_configuration()
            .pubsub_broker,
            actor_model_path=os.path.join(self.__base_path, f"predator_{os.environ.get('REL_PATH')}.keras"),
            routing_key="predator-actor-model",
        )

    def prey_policy_controller_simulation(self) -> AgentPolicyController:
        return PredatorPreyPolicyControllerSimulation(
            actor_model_path=os.path.join(self.__base_path, f"prey_{os.environ.get('REL_PATH')}.keras"),
        )

    def predator_policy_controller_simulation(self) -> AgentPolicyController:
        return PredatorPreyPolicyControllerSimulation(
            actor_model_path=os.path.join(self.__base_path, f"predator_{os.environ.get('REL_PATH')}.keras"),
        )
