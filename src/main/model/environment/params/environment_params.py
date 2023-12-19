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


class EnvironmentParamsFactory:

    @staticmethod
    def standard_parameters() -> EnvironmentParams:
        return EnvironmentParams(
            x_dim=500,
            y_dim=500,
            num_predators=10,
            num_preys=10,
            num_states=14,
            num_actions=2,
            lower_bound=-1,
            upper_bound=1,
            r=1,
            vd=30,
            life=100
        )
