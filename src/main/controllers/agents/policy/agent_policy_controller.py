class AgentPolicyController:
    def policy(self, state):
        """
        Base reward method, to be overridden by subclasses
        :return: action
        """
        raise NotImplementedError("Subclasses must implement this method")

    def stop(self):
        """
        Base Stop method to stop update policy, to be overridden by subclasses
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")
