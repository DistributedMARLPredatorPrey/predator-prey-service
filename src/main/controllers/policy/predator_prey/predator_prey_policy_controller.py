from src.main.controllers.policy.predator_prey.actor_receiver_controller import (
    ActorReceiverController,
)
from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class PredatorPreyPolicyController(AgentPolicyController):
    def __init__(self, init: bool, broker_host: str, actor_model_path: str, routing_key: str):
        self.actor_receiver_controller = ActorReceiverController(
            init, broker_host, actor_model_path, routing_key
        )

    def policy(self, state):
        actor = self.actor_receiver_controller.get_latest_actor()
        return actor(state)

    def stop(self):
        self.actor_receiver_controller.stop_recv = True
        self.actor_receiver_controller.recv_thread.join()
