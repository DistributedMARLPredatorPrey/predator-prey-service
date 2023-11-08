import unittest

from main.controllers.environment.observation import is_eating
from main.model.agents.predator import Predator
from main.model.agents.prey import Prey


class TestEating(unittest.TestCase):

    def test_simple_eating(self):
        predator = Predator(id="predator-1", x=2, y=2)
        prey = Prey(id="predator-1", x=2, y=2)
        self.assertTrue(is_eating(predator, prey, r=0.5))

    def test_horizontal_eating(self):
        predator = Predator(id="predator-1", x=1.5, y=1.5)
        prey = Prey(id="predator-1", x=2, y=2)
        self.assertTrue(is_eating(predator, prey, r=0.5))

    def test_vertical_eating(self):
        predator = Predator(id="predator-1", x=1.5, y=2.5)
        prey = Prey(id="predator-1", x=2, y=2)
        self.assertTrue(is_eating(predator, prey, r=0.5))


if __name__ == '__main__':
    unittest.main()
