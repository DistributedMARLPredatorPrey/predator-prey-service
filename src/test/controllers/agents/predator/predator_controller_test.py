import unittest

from main.controllers.agents.predator_prey.predator import PredatorController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.agents.prey import Prey
from src.main.model.environment.params.environment_params import (
    EnvironmentParamsFactory,
)


class StandardPredatorControllerTest(unittest.TestCase):
    predator_controller = PredatorController(
        EnvironmentParamsFactory.environment_config(),
        Predator("predator-1", 0, 0),
        ParameterService(),
    )

    def test_reward_proportional_to_distance(self):
        self.predator_controller.state([Prey("prey-1", 5, 0)])
        first_reward = self.predator_controller.reward()
        self.predator_controller.state([Prey("prey-1", 10, 0)])
        second_reward = self.predator_controller.reward()
        assert first_reward > second_reward
