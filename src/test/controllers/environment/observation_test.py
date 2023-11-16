import unittest

from src.main.controllers.environment.environment_observer import EnvironmentObserver
from src.main.model.agents.predator import Predator
from src.main.model.agents.prey import Prey


class ObservationTest(unittest.TestCase):

    env_observer = EnvironmentObserver()

    def test_simple_eating(self):
        predator = Predator(id="predator-1", x=2, y=2)
        prey = Prey(id="prey-1", x=2, y=2)
        self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))

    def test_horizontal_eating(self):
        predator = Predator(id="predator-1", x=1.5, y=1.5)
        prey = Prey(id="prey-1", x=2, y=2)
        self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))

    def test_vertical_eating(self):
        predator = Predator(id="predator-1", x=1.5, y=2.5)
        prey = Prey(id="prey-1", x=2, y=2)
        self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))
