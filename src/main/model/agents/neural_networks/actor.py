import tensorflow as tf

from tensorflow.keras import layers


class Actor:

    # The actor choose the move, given the state
    def __init__(self, num_states: int, train_acceleration: bool = True, train_direction: bool = True):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # the actor has separate towers for action and speed
        # in this way we can train them separately
        inputs = layers.Input(shape=(num_states,))
        acc_out = layers.Dense(128, activation="relu")(inputs)
        acc_out = layers.Dense(64, activation="relu")(acc_out)
        # acceleration
        acc_out = layers.Dense(1, activation='tanh', kernel_initializer=last_init)(acc_out)

        ang_acc_out = layers.Dense(128, activation="relu")(inputs)
        ang_acc_out = layers.Dense(64, activation="relu")(ang_acc_out)
        # angular acceleration
        ang_acc_out = layers.Dense(1, activation='tanh', kernel_initializer=last_init)(ang_acc_out)

        outputs = layers.Concatenate()([acc_out, ang_acc_out])

        # outputs = outputs * upper_bound #resize the range, if required
        self.model = tf.keras.Model(inputs, outputs, name="actor")
