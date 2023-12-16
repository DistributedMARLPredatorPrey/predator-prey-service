from src.main.model.environment.buffer.buffer import Buffer


class BufferFactory:

    @staticmethod
    def create_buffers(num_states, num_actions, sizes):
        """
        Creates n buffers where n = |sizes|
        :param num_states: number of states
        :param num_actions: number of actions
        :param sizes: array where each element is the number of agents per type
        :return:
        """
        return [Buffer(50_000, 64, num_states, num_actions, size) for size in sizes]
