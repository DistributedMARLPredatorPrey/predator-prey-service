from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentParams:
    """
    Value object representing environment parameters.
    """
    # Environment parameters
    x_dim: int
    y_dim: int
    num_predators: int
    num_preys: int
    # Agent specific parameters
    num_states: int
    num_actions: int
    lower_bound: float
    upper_bound: float
    r: float
    vd: float
    life: int
    