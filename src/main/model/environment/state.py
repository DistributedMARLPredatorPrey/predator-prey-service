from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class State:
    """
    Value object representing a state
    """

    distances: List[float]
