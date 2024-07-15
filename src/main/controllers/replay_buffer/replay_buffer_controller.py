from typing import Tuple

from src.main.model.environment.agents.agent_type import AgentType


class ReplayBufferController:
    def record(self, agent_type: AgentType, record_tuple: Tuple):
        """
        Base record method, to be overridden by subclasses
        :param agent_type: agent type
        :param record_tuple: tuple
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")
