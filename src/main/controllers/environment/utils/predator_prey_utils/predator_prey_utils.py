from functools import partial
from multiprocessing import Pool

from src.main.controllers.agents.policy.agent_policy_controller_factory import (
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
    def initialize_policy_receivers(project_root_path: str):
        """
        Initialize policy receivers of both predator and prey.
        """
        partial_prey_policy_controller = partial(
            AgentPolicyControllerFactory(
                project_root_path=project_root_path
            ).prey_policy_controller_learning,
            init=True,
        )
        partial_pred_policy_controller = partial(
            AgentPolicyControllerFactory(
                project_root_path=project_root_path
            ).predator_policy_controller_learning,
            init=True,
        )
        with Pool(2) as p:
            p.map(
                PredatorPreyUtils.call_f,
                [partial_prey_policy_controller, partial_pred_policy_controller],
            )
