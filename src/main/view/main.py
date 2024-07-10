import logging
from datetime import datetime

from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.controllers.environment.environment_controller_factory import (
    EnvironmentControllerFactory,
)

if __name__ == "__main__":
    init = True
    while True:
        logging.info("Starting...")
        env_controller: EnvironmentController = (
            EnvironmentControllerFactory().create_predator_prey(init=init)
        )
        env_controller.train()
        init = False