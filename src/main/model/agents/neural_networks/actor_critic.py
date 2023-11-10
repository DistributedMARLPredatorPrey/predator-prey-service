from tensorflow.keras import layers, Model

from main.model.agents.neural_networks.actor import Actor
from main.model.agents.neural_networks.critic import Critic


class ActorCritic:

    def __init__(self, num_states, num_actions, train_acceleration=True, train_direction=True):
        self.num_states = num_states
        self.num_actions = num_actions
        self.model = self._compose(
            Actor(num_states, train_acceleration, train_direction).model,
            Critic(num_states, num_actions).model
        )

    # We compose actor and critic in a single model.
    # The actor is trained by maximizing the future expected reward, estimated
    # by the critic. The critic should be frozen while training the actor.
    # For simplicity, we just use the target critic, that is not trainable.
    def _compose(self, actor: Model, critic: Model) -> Model:
        state_input = layers.Input(shape=self.num_states)
        a = actor(state_input)
        q = critic([state_input, a])
        model = Model(state_input, q)
        # the loss function of the compound model is just the opposite of the critic output
        model.add_loss(-q)
        return model
