
class ReplayBufferController:

    def record(self, record_tuple):
        """
        Base record method, to be overridden by subclasses
        :return: action
        """
        raise NotImplementedError("Subclasses must implement this method")