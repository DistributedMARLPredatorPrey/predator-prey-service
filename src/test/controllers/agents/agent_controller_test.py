import unittest

from src.main.controllers.agents.agent_controller import AgentController
from src.main.controllers.parameter_server.parameter_service import ParameterService
from src.main.model.agents.predator import Predator
from src.main.model.agents.prey import Prey
from src.main.model.environment.params.environment_params import EnvironmentParamsFactory
from src.main.model.environment.state import State


class StandardAgentControllerTest(unittest.TestCase):
    std_parameters = EnvironmentParamsFactory.standard_parameters()

    agent_controller = AgentController(
        std_parameters,
        Prey("prey-1", 0, 0),
        ParameterService()
    )

    def test_state_size_as_env_parameters(self):
        match self.agent_controller.state([]):
            case State(distances):
                assert len(distances) == self.std_parameters.num_states

    def test_state_maximum_visual_depth_constraint(self):
        assert (self.agent_controller.state([]) ==
                State([self.std_parameters.vd for _ in range(self.std_parameters.num_states)]))

    def test_state_two_agents_observation(self):
        match self.agent_controller.state([Predator("predator-1", -10, 0),
                                           Predator("predator-2", 10, 0)]):
            case State(distances):
                assert len([distance for distance in distances if distance < self.std_parameters.vd]) == 2

    def test_prey_eaten_by_predator(self):
        assert self.agent_controller.is_eaten(Predator("predator-1", 0, 0))
