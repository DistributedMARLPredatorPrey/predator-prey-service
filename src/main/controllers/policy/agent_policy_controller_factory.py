from src.main.controllers.policy.predator_prey.predator_prey_policy_controller import (
    PredatorPreyPolicyController,
)
from src.main.model.config.config_utils import ConfigUtils
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class AgentPolicyControllerFactory:
    @staticmethod
    def prey_policy_controller() -> AgentPolicyController:
        return PredatorPreyPolicyController(
            broker_host=ConfigUtils().learner_service_configuration().pubsub_broker,
            actor_model_path="src/main/resources/prey.keras",
            routing_key="prey-actor-model",
        )

    @staticmethod
    def predator_policy_controller() -> AgentPolicyController:
        return PredatorPreyPolicyController(
            broker_host=ConfigUtils().learner_service_configuration().pubsub_broker,
            actor_model_path="src/main/resources/predator.keras",
            routing_key="predator-actor-model",
        )
