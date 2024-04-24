class AgentPolicyController:

    def policy(self, state):
        """
        Base reward method, to be overridden by subclasses
        :return: action
        """
        raise NotImplementedError("Subclasses must implement this method")
