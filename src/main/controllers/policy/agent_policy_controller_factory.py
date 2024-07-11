import os

from src.main.controllers.policy.predator_prey.predator_prey_policy_controller import (
    PredatorPreyPolicyController,
)
from src.main.model.config.config_utils import ConfigUtils
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class AgentPolicyControllerFactory:
    @staticmethod
    def prey_policy_controller(init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyController(
            init=init,
            broker_host=ConfigUtils().learner_service_configuration().pubsub_broker,
            actor_model_path=f"src/main/resources/prey_{os.environ.get('REL_PATH')}.keras",
            routing_key="prey-actor-model",
        )

    @staticmethod
    def predator_policy_controller(init: bool) -> AgentPolicyController:
        return PredatorPreyPolicyController(
            init=init,
            broker_host=ConfigUtils().learner_service_configuration().pubsub_broker,
            actor_model_path=f"src/main/resources/predator_{os.environ.get('REL_PATH')}.keras",
            routing_key="predator-actor-model",
        )
