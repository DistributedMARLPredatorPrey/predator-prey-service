from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Value object representing environment config.
    """
    # Environment parameters
    x_dim: int
    y_dim: int
    num_predators: int
    num_preys: int
    # Agent specific parameters
    num_states: int
    num_actions: int
    acc_lower_bound: float
    acc_upper_bound: float
    r: float
    vd: float
    life: int

