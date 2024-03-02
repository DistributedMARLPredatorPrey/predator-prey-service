from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Sum type modelling the environment configuration.
    """
    x_dim: int
    y_dim: int
    num_predators: int
    num_preys: int
    num_states: int
    num_actions: int
    acc_lower_bound: float
    acc_upper_bound: float
    r: float
    vd: float
    life: int


@dataclass(frozen=True)
class ReplayBufferServiceConfig:
    """
    Sum type modelling the replay buffer configuration.
    """
    replay_buffer_host: str
    replay_buffer_port: int


@dataclass
class LearnerServiceConfig:
    """
    Sum type modelling the lerner service configuration.
    """
    pubsub_broker: str
