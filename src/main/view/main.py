from datetime import datetime

from src.main.controllers.environment.environment_controller import EnvironmentController
from src.main.controllers.environment.environment_controller_factory import EnvironmentControllerFactory

if __name__ == '__main__':
    env_controller: EnvironmentController = (EnvironmentControllerFactory()
                                             .create_random())
    start_t = datetime.now()
    env_controller.train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t - start_t))
