import numpy as np

from view.tracks import Racer


class Environment:
    def __init__(self, x_dim=None, y_dim=None):
        (self.x_dim, self.y_dim) = (500, 500) if (x_dim is None or y_dim is None) else (x_dim, y_dim)
        self.num_states = 5
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1

        # Define the racer object. This is the main class of the library.
        self.racer = Racer(obstacles=True,
                      turn_limit=True,
                      chicanes=True,
                      low_speed_termination=True)

    # We introduce a probability of doing n empty actions to separate the environment time-step from the agent
    def step(self, action):
        n = 1
        t = np.random.randint(0, n)
        state, reward, done = self.racer.step(action)
        for i in range(t):
            if not done:
                state, t_r, done = self.racer.step([0, 0])
                # state ,t_r, done =racer.step(action)
                reward += t_r
        return state, reward, done