from tensorflow.keras.models import load_model

from src.main.controllers.policy.agent_policy_controller import AgentPolicyController


class PredatorPreyPolicyControllerSimulation(AgentPolicyController):
    def __init__(self, actor_model_path: str):
        self.__actor = load_model(actor_model_path)

    def policy(self, state):
        return self.__actor(state)

    def stop(self):
        pass
