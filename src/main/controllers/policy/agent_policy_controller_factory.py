from main.controllers.policy.predator_prey.predator_prey_policy_controller import (
    PredatorPreyPolicyController,
)
from main.model.config.config_utils import ConfigUtils
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class AgentPolicyControllerFactory:
    @staticmethod
    def prey_policy_controller() -> AgentPolicyController:
        return PredatorPreyPolicyController(
            broker_host=ConfigUtils().learner_service_configuration(),
            actor_model_path="src/main/resources/prey.h5",
            routing_key="prey-actor-model",
        )

    @staticmethod
    def predator_policy_controller() -> AgentPolicyController:
        return PredatorPreyPolicyController(
            broker_host=ConfigUtils().learner_service_configuration(),
            actor_model_path="src/main/resources/predator.h5",
            routing_key="predator-actor-model",
        )
