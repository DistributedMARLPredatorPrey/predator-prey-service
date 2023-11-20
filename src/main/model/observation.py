from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Observation:
    state: List[float]
    done: bool
    reward: int
