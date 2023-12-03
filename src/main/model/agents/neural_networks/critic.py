import tensorflow as tf

from tensorflow.keras.layers import *

class Critic:

    # The critic compute the q-value, given the state and the action
    def __init__(self, num_states: int, num_actions: int, num_agents: int):
        # State as input
        state_input = Input(shape=(num_states * num_agents,))
        state_out = Dense(128, activation="relu")(state_input)
        state_out = Dense(256, activation="relu")(state_out)

        # Action as input
        action_input = Input(shape=(num_actions * num_agents,))
        action_out = Dense(256, activation="relu")(action_input)

        concat = Concatenate()([state_out, action_out])

        out = Dense(128, activation="relu")(concat)
        out = Dense(128, activation="relu")(out)
        outputs = Dense(1)(out)  # Outputs single value
        self.model = tf.keras.Model([state_input, action_input], outputs, name="critic")
