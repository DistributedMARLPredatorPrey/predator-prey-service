import requests

from src.main.model.agents.agent_type import AgentType


class ReplayBufferController:

    def __init__(self):
        self._ip = "172.17.0.2"
        self._port = 80

    def record(self, record_tuples):
        self._send_prey_record_tuple(record_tuples[AgentType.PREY])
        self._send_predator_record_tuple(record_tuples[AgentType.PREDATOR])

    def _send_prey_record_tuple(self, record_tuple):
        self._send_record_tuple("record_data/prey/", record_tuple)

    def _send_predator_record_tuple(self, record_tuple):
        self._send_record_tuple("record_data/predator/", record_tuple)

    def _send_record_tuple(self, route, record_tuple):
        prev_states, actions, rewards, next_states = record_tuple
        record_json = {
            "State": prev_states,
            "Reward": actions,
            "Action": rewards,
            "Next state": next_states
        }
        requests.post(f"http://{self._ip}:{self._port}/{route}", record_json)
