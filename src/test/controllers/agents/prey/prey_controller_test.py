import unittest

import pytest

from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.agents.prey import Prey
from src.main.model.environment.params.environment_params import EnvironmentParamsFactory
from src.main.controllers.agents.prey.prey_controller import PreyController


class PreyControllerTest(unittest.TestCase):
    prey_controller = PreyController(
        EnvironmentParamsFactory.standard_parameters(),
        Prey("prey-1", 0, 0),
        ParameterService()
    )

    @pytest.mark.description(
        "The prey should receive a reward which is proportional to the distance of the closest predator")
    def test_reward(self):
        self.prey_controller.state([Predator("predator-1", 5, 0)])
        first_reward = self.prey_controller.reward()
        self.prey_controller.state([Predator("predator-1", 10, 0)])
        second_reward = self.prey_controller.reward()
        assert first_reward < second_reward
