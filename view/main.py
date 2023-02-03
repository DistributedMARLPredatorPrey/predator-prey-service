from controllers.predator_controller import PredatorController
import tracks
from model.environment import Environment
from model.predator import Predator

"""
if total_iterations > 0:
    if save_weights:
        critic_model.save(weights_file_critic)
        actor_model.save(weights_file_actor)
    # Plotting Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Training steps x100")
    plt.ylabel("Avg. Episodic Reward")
    plt.ylim(-3.5, 7)
    plt.show(block=False)
    plt.pause(0.001)
    print("### DDPG Training ended ###")
    print("Trained over {} steps".format(i))
"""

if __name__ == '__main__':
    n_agents = 3
    save_weights = True
    env = Environment()
    agents_controller = [PredatorController(env=env, predator=Predator(i)) for i in range(n_agents)]

    for i in range(1, 1500):
        for ac in agents_controller:
            ac.iterate(i)

    if save_weights:
        for ac in agents_controller:
            ac.save()

    tracks.newrun([ac.actor_model for ac in agents_controller])
