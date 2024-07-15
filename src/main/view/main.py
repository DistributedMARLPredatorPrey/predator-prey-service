import logging

import numpy as np

from src.main.model.config.config import EnvironmentConfig, Mode
from src.main.model.config.config_utils import PredatorPreyConfig
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.controllers.environment.environment_controller_factory import (
    EnvironmentControllerFactory,
)

logging.getLogger().setLevel(logging.INFO)
import random


def train():
    """
    Run Predator Prey Service in Training mode
    :return:
    """
    init = True
    while True:
        logging.info("Starting Predator-Prey Training...")
        env_controller: EnvironmentController = (
            EnvironmentControllerFactory().create_predator_prey_learning(
                init=init, pred_prey_config=PredatorPreyConfig()
            )
        )
        env_controller.train()
        init = False


def simulate():
    """
    Run Predator Prey Service in Simulation mode
    :return:
    """
    predator_prey_config = PredatorPreyConfig()
    # Set seed for reproducibility
    random.seed(predator_prey_config.environment_configuration().random_seed)
    while True:
        logging.info("Starting Predator-Prey Simulation...")
        env_controller: EnvironmentController = (
            EnvironmentControllerFactory().create_predator_prey_simulation(
                pred_prey_config=predator_prey_config
            )
        )
        env_controller.simulate()


if __name__ == "__main__":
    config: EnvironmentConfig = PredatorPreyConfig().environment_configuration()
    train() if config.mode == Mode.TRAINING else simulate()
