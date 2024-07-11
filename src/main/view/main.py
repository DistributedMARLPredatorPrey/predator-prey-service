import logging

from src.main.model.config.config import EnvironmentConfig, Mode
from src.main.model.config.config_utils import ConfigUtils
from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.controllers.environment.environment_controller_factory import (
    EnvironmentControllerFactory,
)

logging.getLogger().setLevel(logging.INFO)


def train():
    """
    Run Predator Prey Service in Training mode
    :return:
    """
    init = True
    while True:
        logging.info("Starting Predator-Prey Training...")
        env_controller: EnvironmentController = (
            EnvironmentControllerFactory().create_predator_prey_learning(init=init)
        )
        env_controller.train()
        init = False


def simulate():
    """
    Run Predator Prey Service in Simulation mode
    :return:
    """
    while True:
        logging.info("Starting Predator-Prey Simulation...")
        env_controller: EnvironmentController = (
            EnvironmentControllerFactory().create_predator_prey_simulation()
        )
        env_controller.simulate()


if __name__ == "__main__":
    config: EnvironmentConfig = ConfigUtils().environment_configuration()
    train() if config.mode == Mode.TRAINING else simulate()
