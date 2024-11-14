from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    """
    Enum modelling the possible ways to run the service
    """

    TRAINING = 1
    SIMULATION = 2


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
    r: float
    vd: float
    life: int
    save_experiment_data: bool
    base_experiment_path: str
    rel_experiment_path: str
    mode: Mode
    random_seed: int


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
