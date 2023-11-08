import unittest

from main.controllers.environment.environment_observer import EnvironmentObserver
from main.model.agents.predator import Predator
from main.model.agents.prey import Prey
from main.model.environment import Environment


class ObservationTest(unittest.TestCase):

    env_observer = EnvironmentObserver()

    def test_simple_observation(self):
        coordinates = [(10, 10), (10, 8), (10, 12), (8, 10), (10, 8)]
        env: Environment = Environment(agents=[Predator("%d".format(i), x, y) for i, (x, y) in enumerate(coordinates)])
        self.assertEqual(self.env_observer.observe(env.agents[0], env),
                         ([1.7, -1, -1, 1.7, -1, -1, 1.7, -1, -1, -1, 1.7, -1, -1, -1], 1))

    def test_simple_eating(self):
        predator = Predator(id="predator-1", x=2, y=2)
        prey = Prey(id="predator-1", x=2, y=2)
        self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))

    def test_horizontal_eating(self):
        predator = Predator(id="predator-1", x=1.5, y=1.5)
        prey = Prey(id="predator-1", x=2, y=2)
        self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))

    def test_vertical_eating(self):
        predator = Predator(id="predator-1", x=1.5, y=2.5)
        prey = Prey(id="predator-1", x=2, y=2)
        self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))
