import json
import logging

import requests

from src.main.controllers.replay_buffer.replay_buffer_controller import ReplayBufferController


class RemoteReplayBufferController(ReplayBufferController):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

    def record(self, record_tuple):
        """
        Record a tuple to a remote Replay Buffer Service
        :param record_tuple: tuple to record
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
            f"http://{self._host}:{self._port}/record_data/predator/", json=json_data
        )
