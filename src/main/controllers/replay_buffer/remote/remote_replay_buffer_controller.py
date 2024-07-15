import json
import logging
from typing import Tuple

import requests

from src.main.model.environment.agents.agent_type import AgentType
from src.main.controllers.replay_buffer.replay_buffer_controller import (
    ReplayBufferController,
)


class RemoteReplayBufferController(ReplayBufferController):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

    def record(self, agent_type: AgentType, record_tuple: Tuple):
        """
        Record a tuple to a remote Replay Buffer Service
        :param agent_type: agent type
        :param record_tuple: tuple
        :return:
        """
        prev_states, actions, rewards, next_states = record_tuple
        record_json = {
            "State": [[float(ps) for prev_state in prev_states for ps in prev_state]],
            "Reward": [[float(r) for r in rewards]],
            "Action": [[float(ps) for action in actions for ps in action]],
            "Next state": [
                [float(ns) for next_state in next_states for ns in next_state]
            ],
        }
        json_data = json.dumps(record_json)
        logging.info("Posting data to replay buffer")
        requests.post(
            f"http://{self._host}:{self._port}/record_data/"
            f"{'predator' if agent_type == AgentType.PREDATOR else 'prey'}/",
            json=json_data
        )
