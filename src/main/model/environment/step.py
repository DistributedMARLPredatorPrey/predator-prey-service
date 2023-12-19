from dataclasses import dataclass

from src.main.model.environment.state import State


@dataclass(frozen=True)
class Step:
    state: State
    done: bool
    reward: int
