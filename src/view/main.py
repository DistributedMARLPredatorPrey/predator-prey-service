import keras

from controllers.predator_controller import PredatorController
from controllers.environment_controller import Environment
from model.predator import Predator
from datetime import datetime


def train():
    # parameters
    n_agents = 3
    total_iterations = 40_000

    save_weights = True
    load_weights = False

    # environment
    env = Environment()
    # controllers
    if load_weights:
        agents_controller = []
        for i in range(n_agents):
            actor_model = keras.models \
                .load_model('./predatormodel/{agent_id}/actormodel'.format(agent_id=i))
            critic_model = keras.models \
                .load_model('./predatormodel/{agent_id}/criticmodel'.format(agent_id=i))
            agents_controller.append(PredatorController(env=env,
                                                        predator=Predator(i),
                                                        actor_model=actor_model,
                                                        critic_model=critic_model))
    else:
        agents_controller = [PredatorController(env=env, predator=Predator(i)) for i in range(n_agents)]

    # train
    # TODO
    # ac_it = [0 for _ in range(n_agents)]
    # while any(it < total_iterations for it in ac_it):
    #    for k in range(n_agents):
    #        if ac_it[k] < total_iterations:
    #           ac_it[k] = agents_controller[k].iterate(ac_it[k])

    # if save_weights:
    #    for ac in agents_controller:
    #        ac.save()

    # tracks.newrun([ac.actor_model for ac in agents_controller])


if __name__ == '__main__':
    start_t = datetime.now()
    train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t - start_t))
