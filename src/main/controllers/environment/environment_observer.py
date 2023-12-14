from typing import List

import numpy as np
from z3 import Or, And, If, Solver, Optimize, AlgebraicNumRef, sat, Real

from src.main.model.agents.agent import Agent
from src.main.model.environment.environment import Environment
from src.main.model.environment.observation import Observation

np.random.seed(42)


class EnvironmentObserver:

    def __init__(self, r: float = 5, vd: float = 30):
        self.r = r
        self.vd = vd


