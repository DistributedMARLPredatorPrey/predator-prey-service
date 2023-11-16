import tensorflow as tf


class Actor:

    # The actor choose the move, given the state
    def __init__(self, num_states: int, train_acceleration: bool = True, train_direction: bool = True):
        # the actor has separate towers for action and speed
        # in this way we can train them separately
        inputs = tf.keras.layers.Input(shape=(num_states,))
        out1 = tf.keras.layers.Dense(32, activation="relu", trainable=train_acceleration)(inputs)
        out1 = tf.keras.layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
        # acceleration
        out1 = tf.keras.layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)

        out2 = tf.keras.layers.Dense(32, activation="relu", trainable=train_direction)(inputs)
        out2 = tf.keras.layers.Dense(32, activation="relu", trainable=train_direction)(out2)
        # angular acceleration
        out2 = tf.keras.layers.Dense(1, activation='tanh', trainable=train_direction)(out2)

        outputs = tf.keras.layers.concatenate([out1, out2])

        # outputs = outputs * upper_bound #resize the range, if required
        self.model = tf.keras.Model(inputs, outputs, name="actor")


