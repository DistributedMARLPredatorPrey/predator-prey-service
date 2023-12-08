import tensorflow as tf

from tensorflow.keras import layers

class Critic:

    # The critic compute the q-value, given the state and the action
    def __init__(self, id, num_states: int, num_actions: int, num_agents: int):
        # State as input
        state_input = layers.Input(shape=(num_states * num_agents,), name=f"p{id}_input")
        state_out = layers.Dense(128, activation="relu", name=f"p{id}_1")(state_input)
        state_out = layers.Dense(256, activation="relu", name=f"p{id}_2")(state_out)

        # Action as input
        action_input = [layers.Input(shape=num_actions, name=f"{id}_i{k}") for k in range(num_agents)]
        action_input_concat = layers.Concatenate(name=f"{id}_c")(action_input)
        action_out = layers.Dense(256, activation="relu", name=f"p{id}_3")(action_input_concat)

        concat = layers.Concatenate(name=f"{id}_4")([state_out, action_out])

        out = layers.Dense(128, activation="relu", name=f"p{id}_5")(concat)
        out = layers.Dense(128, activation="relu", name=f"p{id}_6")(out)
        outputs = layers.Dense(1)(out)  # Outputs single value
        self.model = tf.keras.Model([state_input, action_input], outputs, name="critic")

        #print(self.model.summary())
