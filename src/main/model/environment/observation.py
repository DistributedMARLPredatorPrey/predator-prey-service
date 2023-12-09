from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Observation:
    observation: List[float]
