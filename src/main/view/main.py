from datetime import datetime

from src.main.controllers.environment.environment_controller import (
    EnvironmentController,
)
from src.main.controllers.environment.environment_controller_factory import (
    EnvironmentControllerFactory,
)

if __name__ == "__main__":
    print("Pred-Prey-Service: starting")
    start_t = datetime.now()
    while True:
        env_controller: EnvironmentController = (
            EnvironmentControllerFactory().create_predator_prey()
        )
        env_controller.train()
        print("Done")
