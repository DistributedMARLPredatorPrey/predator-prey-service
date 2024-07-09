import unittest

# from src.main.controllers.agents.predator_prey.predator.predator_controller_factory import PredatorControllerFactory
# from src.main.model.agents.predator import Predator
# from src.main.controllers.agents.predator_prey.predator.predator_controller import PredatorController
# from src.main.model.config.config_utils import ConfigUtils
#

# class StandardPredatorControllerTest(unittest.TestCase):
#
#
#
#     predator_controllers = PredatorControllerFactory.create_from_params(
#         self._env_config, self._pred_actor_receiver_controller
#     )

# predator_controller = PredatorController(
#     EnvironmentParamsFactory.environment_config(),
#     Predator("predator-1", 0, 0),
#     ParameterService(),
# )

# def test_reward_proportional_to_distance(self):
#     self.predator_controller.state([Prey("prey-1", 5, 0)])
#     first_reward = self.predator_controller.reward()
#     self.predator_controller.state([Prey("prey-1", 10, 0)])
#     second_reward = self.predator_controller.reward()
#     assert first_reward > second_reward
