import os

from src.main.controllers.policy.predator_prey.predator_prey_policy_controller_simulation import (
    PredatorPreyPolicyControllerSimulation,
)
from src.main.controllers.policy.predator_prey.predator_prey_policy_controller_learning import (
    PredatorPreyPolicyControllerLearning,
)
from src.main.model.config.config_utils import PredatorPreyConfig
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class AgentPolicyControllerFactory:
    @staticmethod
    def prey_policy_controller_learning(init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyControllerLearning(
            init=init,
            broker_host=PredatorPreyConfig()
            .learner_service_configuration()
            .pubsub_broker,
            actor_model_path=f"src/main/resources/prey_{os.environ.get('REL_PATH')}.keras",
            routing_key="prey-actor-model",
        )

    @staticmethod
    def predator_policy_controller_learning(init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyControllerLearning(
            init=init,
            broker_host=PredatorPreyConfig()
            .learner_service_configuration()
            .pubsub_broker,
            actor_model_path=f"src/main/resources/predator_{os.environ.get('REL_PATH')}.keras",
            routing_key="predator-actor-model",
        )

    @staticmethod
    def prey_policy_controller_simulation() -> AgentPolicyController:
        return PredatorPreyPolicyControllerSimulation(
            actor_model_path=f"src/main/resources/prey_{os.environ.get('REL_PATH')}.keras",
        )

    @staticmethod
    def predator_policy_controller_simulation() -> AgentPolicyController:
        return PredatorPreyPolicyControllerSimulation(
            actor_model_path=f"src/main/resources/predator_{os.environ.get('REL_PATH')}.keras",
        )
