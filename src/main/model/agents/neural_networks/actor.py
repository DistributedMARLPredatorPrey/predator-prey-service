import tensorflow as tf

from tensorflow.keras import layers


class Actor:

    # The actor choose the move, given the state
    def __init__(self, id, num_states: int, train_acceleration: bool = True, train_direction: bool = True):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # the actor has separate towers for action and speed
        # in this way we can train them separately
        inputs = layers.Input(shape=(num_states,), name=f"a{id}_input")
        acc_out = layers.Dense(128, activation="relu", name=f"a{id}_1")(inputs)
        acc_out = layers.Dense(64, activation="relu", name=f"a{id}_2")(acc_out)
        # acceleration
        acc_out = layers.Dense(1, activation='tanh', kernel_initializer=last_init, name=f"a{id}_3")(acc_out)

        ang_acc_out = layers.Dense(128, activation="relu", name=f"a{id}_4")(inputs)
        ang_acc_out = layers.Dense(64, activation="relu", name=f"a{id}_5")(ang_acc_out)
        # angular acceleration
        ang_acc_out = layers.Dense(1, activation='tanh', kernel_initializer=last_init, name=f"a{id}_6")(ang_acc_out)

        outputs = layers.Concatenate(name=f"a{id}_7")([acc_out, ang_acc_out])

        # outputs = outputs * upper_bound #resize the range, if required
        self.model = tf.keras.Model(inputs, outputs, name="actor")
