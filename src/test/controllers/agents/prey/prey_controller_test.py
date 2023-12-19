import unittest

from src.main.controllers.agents.prey.prey_controller import PreyController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.agents.prey import Prey
from src.main.model.environment.params.environment_params import EnvironmentParamsFactory


class StandardPreyControllerTest(unittest.TestCase):
    prey_controller = PreyController(
        EnvironmentParamsFactory.standard_parameters(),
        Prey("prey-1", 0, 0),
        ParameterService()
    )

    def test_reward_inversely_proportional_to_distance(self):
        self.prey_controller.state([Predator("predator-1", 5, 0)])
        first_reward = self.prey_controller.reward()
        self.prey_controller.state([Predator("predator-1", 10, 0)])
        second_reward = self.prey_controller.reward()
        assert first_reward < second_reward
