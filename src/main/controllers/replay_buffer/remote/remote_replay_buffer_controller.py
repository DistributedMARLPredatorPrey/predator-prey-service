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

    def record(self, record_tuple: Tuple):
        """
        Record a tuple to a remote Replay Buffer Service
        :param record_tuple: tuple
        :return:
        """
        prev_states, actions, rewards, next_states = record_tuple
        record_json = {
            "State": [[float(ps) for prev_state in prev_states for ps in prev_state]],
            "Reward": [[float(r) for r in rewards]],
            "Action": [[float(ps) for action in actions for ps in action]],
            "Next State": [
                [float(ns) for next_state in next_states for ns in next_state]
            ],
        }
        requests.post(
            f"http://{self._host}:{self._port}/record_data/",
            json=record_json,
        )
