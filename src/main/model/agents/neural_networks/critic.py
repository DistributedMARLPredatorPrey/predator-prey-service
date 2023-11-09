from tensorflow.keras import layers, Model


class Critic:

    # The critic compute the q-value, given the state and the action
    def __init__(self, num_states, num_actions):
        # State as input
        state_input = layers.Input(shape=num_states)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=num_actions)
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1)(out)  # Outputs single value
        self.model = Model([state_input, action_input], outputs, name="critic")
