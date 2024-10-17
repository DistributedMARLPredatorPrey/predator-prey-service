from typing import Tuple


class ReplayBufferController:
    def record(self, record_tuple: Tuple):
        """
        Base record method, to be overridden by subclasses
        :param record_tuple: tuple
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")
