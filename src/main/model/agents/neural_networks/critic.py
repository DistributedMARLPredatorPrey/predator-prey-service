import tensorflow as tf

from tensorflow.keras import layers

class Critic:

    # The critic compute the q-value, given the state and the action
    def __init__(self, num_states: int, num_actions: int, num_agents: int):
        # State as input
        state_input = layers.Input(shape=(num_states * num_agents,))
        state_out = layers.Dense(128, activation="relu")(state_input)
        state_out = layers.Dense(256, activation="relu")(state_out)

        # Action as input
        action_input = [layers.Input(shape=num_actions) for _ in range(num_agents)]
        action_input_concat = layers.Concatenate()(action_input)
        action_out = layers.Dense(256, activation="relu",)(action_input_concat)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(128, activation="relu")(out)
        outputs = layers.Dense(1)(out)  # Outputs single value
        self.model = tf.keras.Model([state_input, action_input], outputs, name="critic")
