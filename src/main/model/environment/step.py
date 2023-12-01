from dataclasses import dataclass

from src.main.model.environment.observation import Observation


@dataclass(frozen=True)
class Step:
    observation: Observation
    done: bool
    reward: int
