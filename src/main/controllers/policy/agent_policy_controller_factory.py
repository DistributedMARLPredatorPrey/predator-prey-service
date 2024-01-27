import os

from main.controllers.policy.predator_prey.predator_prey_policy_controller import (
    PredatorPreyPolicyController,
)
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class AgentPolicyControllerFactory:
    @staticmethod
    def prey_policy_controller() -> AgentPolicyController:
        return PredatorPreyPolicyController(
            actor_model_path=os.environ.get("PREY_ACTOR_PATH"),
            routing_key="prey-actor-model",
        )

    @staticmethod
    def predator_policy_controller() -> AgentPolicyController:
        return PredatorPreyPolicyController(
            actor_model_path=os.environ.get("PREDATOR_ACTOR_PATH"),
            routing_key="predator-actor-model",
        )
