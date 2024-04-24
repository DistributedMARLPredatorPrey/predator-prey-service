import unittest


#class StandardPreyControllerTest(unittest.TestCase):
# prey_controller = PreyController(
#     EnvironmentParamsFactory.environment_config(),
#     Prey("prey-1", 0, 0),
#     ParameterService(),
# )
#
# def test_reward_inversely_proportional_to_distance(self):
#     self.prey_controller.state([Predator("predator-1", 5, 0)])
#     first_reward = self.prey_controller.reward()
#     self.prey_controller.state([Predator("predator-1", 10, 0)])
#     second_reward = self.prey_controller.reward()
#     assert first_reward < second_reward
