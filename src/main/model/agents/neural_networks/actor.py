from tensorflow.keras import layers, Model


class Actor:

    # The actor choose the move, given the state
    def __init__(self, num_states: int, train_acceleration: bool = True, train_direction: bool = True):
        # the actor has separate towers for action and speed
        # in this way we can train them separately
        inputs = layers.Input(shape=(num_states,))
        out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(inputs)
        out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
        # acceleration
        out1 = layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)

        out2 = layers.Dense(32, activation="relu", trainable=train_direction)(inputs)
        out2 = layers.Dense(32, activation="relu", trainable=train_direction)(out2)
        # angular acceleration
        out2 = layers.Dense(1, activation='tanh', trainable=train_direction)(out2)

        outputs = layers.concatenate([out1, out2])

        # outputs = outputs * upper_bound #resize the range, if required
        self.model = Model(inputs, outputs, name="actor")
