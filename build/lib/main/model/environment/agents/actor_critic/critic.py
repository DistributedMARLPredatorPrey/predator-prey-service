import tensorflow as tf

from tensorflow.keras import layers


class Critic:
    def __init__(self, num_states: int, num_actions: int, num_agents: int):
        """
        The Critic networks is responsible for computing the Q-value,
        given a state and action taken by the Actor network.
        :param num_states: number of states
        :param num_actions: number of actions
        :param num_agents: number of agents inside the environment
        """
        # State as input
        state_input = layers.Input(shape=(num_states * num_agents,))
        state_out = layers.Dense(128, activation="relu")(state_input)
        state_out = layers.Dense(256, activation="relu")(state_out)

        # Action as input
        action_input = [layers.Input(shape=num_actions) for _ in range(num_agents)]
        action_input_concat = layers.Concatenate()(action_input)
        action_out = layers.Dense(
            256,
            activation="relu",
        )(action_input_concat)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(128, activation="relu")(out)
        # Outputs single value
        outputs = layers.Dense(1)(out)
        self.model = tf.keras.Model([state_input, action_input], outputs, name="critic")
