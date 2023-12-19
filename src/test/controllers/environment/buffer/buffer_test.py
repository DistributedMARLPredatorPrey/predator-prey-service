import unittest

import numpy as np
import pytest

from src.main.model.environment.buffer.buffer import Buffer


class BufferTest(unittest.TestCase):
    batch_size, num_states, num_actions, num_agents = 3, 2, 1, 2

    buffer = Buffer(
        batch_size=batch_size,
        num_states=num_states,
        num_actions=num_actions,
        num_agents=num_agents
    )

    def test_batch_size(self):
        self.buffer.record(([0, 0, 0, 0], [0, 0], [0, 0], [0, 0, 0, 0]))
        s, a, r, ns = self.buffer.sample_batch()
        assert (
            np.array(s).shape == (self.batch_size, self.num_states * self.num_agents),
            np.array(a).shape == (self.batch_size, self.num_actions * self.num_agents),
            np.array(r).shape == (self.batch_size, self.num_agents),
            np.array(ns).shape == (self.batch_size, self.num_actions * self.num_agents)
        )
