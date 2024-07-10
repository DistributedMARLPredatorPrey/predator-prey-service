from multiprocessing import Pool

from src.main.controllers.policy.agent_policy_controller_factory import (
    AgentPolicyControllerFactory,
)


class PredatorPreyUtils:
    @staticmethod
    def call_f(f):
        """
        Pick-able function.
        """
        f()

    @staticmethod
    def initialize_policy_receivers():
        """
        Initialize policy receivers of both predator and prey.
        """
        policy_controller_factory = AgentPolicyControllerFactory
        with Pool(2) as p:
            p.map(
                PredatorPreyUtils.call_f,
                [
                    policy_controller_factory.prey_policy_controller,
                    policy_controller_factory.predator_policy_controller,
                ],
            )
