import unittest

import pytest

from main.model.agents.prey import Prey
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.environment.params.environment_params import EnvironmentParamsFactory
from src.main.controllers.agents.predator.predator_controller import PredatorController


class PredatorControllerTest(unittest.TestCase):
    predator_controller = PredatorController(
        EnvironmentParamsFactory.standard_parameters(),
        Predator("predator-1", 0, 0),
        ParameterService()
    )

    @pytest.mark.description(
        "The predator should receive a reward which is inversely proportional to the distance of the closest prey")
    def test_reward(self):
        self.predator_controller.state([Prey("prey-1", 500, 0)])
        first_reward = self.predator_controller.reward()
        self.predator_controller.state([Prey("prey-1", 1000, 0)])
        second_reward = self.predator_controller.reward()
        assert first_reward > second_reward
