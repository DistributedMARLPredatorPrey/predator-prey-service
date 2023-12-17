import unittest

from src.main.controllers.agents.predator.predator_controller import PredatorController
from src.main.model.agents.predator import Predator
from src.main.model.agents.prey import Prey

# class StateTest(unittest.TestCase):

# def test_simple_eating(self):
#     predator = Predator(id="predator-1", x=2, y=2)
#     prey = Prey(id="prey-1", x=2, y=2)
#
#     self.assertTrue(PredatorController())
#
# def test_horizontal_eating(self):
#     predator = Predator(id="predator-1", x=1.5, y=1.5)
#     prey = Prey(id="prey-1", x=2, y=2)
#     self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))
#
# def test_vertical_eating(self):
#     predator = Predator(id="predator-1", x=1.5, y=2.5)
#     prey = Prey(id="prey-1", x=2, y=2)
#     self.assertTrue(self.env_observer.is_eating(predator, prey, r=0.5))
