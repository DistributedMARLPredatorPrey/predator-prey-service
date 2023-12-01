import tensorflow as tf


class Critic:

    # The critic compute the q-value, given the state and the action
    def __init__(self, num_states: int, num_actions: int, num_agents: int):
        # State as input
        state_input = tf.keras.layers.Input(shape=(num_states * num_agents))
        state_out = tf.keras.layers.Dense(128, activation="relu")(state_input)
        state_out = tf.keras.layers.Dense(256, activation="relu")(state_out)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(num_actions * num_agents))
        action_out = tf.keras.layers.Dense(256, activation="relu")(action_input)

        concat = tf.keras.layers.Concatenate()([state_out, action_out])

        out = tf.keras.layers.Dense(128, activation="relu")(concat)
        out = tf.keras.layers.Dense(128, activation="relu")(out)
        outputs = tf.keras.layers.Dense(1)(out)  # Outputs single value
        self.model = tf.keras.Model([state_input, action_input], outputs, name="critic")
