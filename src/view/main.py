import keras

from src.controllers.environment_controller import EnvironmentController
from src.controllers.predator_controller import PredatorController
from src.model.environment import Environment
from src.model.agents.predator import Predator
from datetime import datetime


def train():
    # parameters
    n_agents = 3
    total_iterations = 40_000

    save_weights = True
    load_weights = False

    predators = [Predator(i) for i in range(n_agents)]
    # environment
    env = Environment(x_dim=500, y_dim=500, agents=predators)
    # controllers
    if load_weights:
        agents_controller = []
        for i in range(n_agents):
            actor_model = keras.models \
                .load_model('./predatormodel/{agent_id}/actormodel'.format(agent_id=i))
            critic_model = keras.models \
                .load_model('./predatormodel/{agent_id}/criticmodel'.format(agent_id=i))
            agents_controller.append(PredatorController(env=env,
                                                        predator=predators[i],
                                                        actor_model=actor_model,
                                                        critic_model=critic_model))
    else:
        agents_controller = [PredatorController(env=env, predator=predators[i]) for i in range(n_agents)]

    # train
    # TODO
    #it = 0
    #while it < total_iterations:
    #    for i in range(n_agents):
    #        agents_controller[i].iterate()


    # if save_weights:
    #    for ac in agents_controller:
    #        ac.save()

    # tracks.newrun([ac.actor_model for ac in agents_controller])


if __name__ == '__main__':
    start_t = datetime.now()
    train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t - start_t))
