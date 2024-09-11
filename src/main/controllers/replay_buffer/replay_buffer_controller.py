from typing import Tuple

from src.main.model.environment.agents.agent_type import AgentType


class ReplayBufferController:
    def record(self, record_tuple: Tuple):
        """
        Base record method, to be overridden by subclasses
        :param record_tuple: tuple
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")
