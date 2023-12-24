import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(
        self,
        buffer_capacity=100_000,
        batch_size=64,
        num_states=None,
        num_actions=None,
        num_agents=None,
    ):
        """
        Replay buffer used for batch learning.
        :param buffer_capacity: buffer capacity
        :param batch_size: batch size
        :param num_states: number of states
        :param num_actions: number of actions
        :param num_agents: number of agents
        """
        # Max Number of tuples that can be stored
        self.buffer_capacity = buffer_capacity
        # Num of tuples used for training
        self.batch_size = batch_size

        self.num_states = num_states
        self.num_actions = num_actions

        # Current number of tuples in buffer
        self.buffer_counter = 0

        # We have a different array for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_agents * num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_agents * num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, num_agents))
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, num_agents * num_states)
        )

    def record(self, obs_tuple):
        """
        Records a tuple (s, a, r, s') inside the buffer
        :param obs_tuple: observation tuple
        :return:
        """
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample_batch(self):
        """
        Samples a batch of data from the buffer
        :return: tuple of data
        """
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch
