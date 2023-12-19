import unittest

import pytest

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

    @pytest.mark.description("The state of a prey should have the size specified in the environment params")
    def test_state_size(self):
        match self.agent_controller.state([]):
            case State(distances):
                assert len(distances) == self.std_parameters.num_states

    @pytest.mark.description("The state of a prey should be constrained from the maximum visual depth")
    def test_state_depth(self):
        assert (self.agent_controller.state([]) ==
                State([self.std_parameters.vd for _ in range(self.std_parameters.num_states)]))

    @pytest.mark.description("The prey should be eaten by the predator")
    def test_eat(self):
        assert self.agent_controller.is_eaten(Predator("predator-1", 0, 0))
