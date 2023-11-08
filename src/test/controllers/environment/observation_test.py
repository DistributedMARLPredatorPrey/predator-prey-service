import unittest

from main.controllers.environment.observation import observe
from main.model.agents.predator import Predator
from main.model.environment import Environment


class ObservationTest(unittest.TestCase):

    def test_simple_observation(self):
        coordinates = [(10, 10), (10, 8), (10, 12), (8, 10), (10, 8)]
        env: Environment = Environment(agents=[Predator("%d".format(i), x, y) for i, (x, y) in enumerate(coordinates)])
        self.assertEqual(observe(env.agents[0], env),
                         ([1.7, -1, -1, 1.7, -1, -1, 1.7, -1, -1, -1, 1.7, -1, -1, -1], 1))
