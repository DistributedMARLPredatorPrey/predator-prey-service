import os
import time
import pandas as pd


class EnvironmentControllerUtils:

    def __init__(self, base_experiment_path, rel_experiment_path):
        self.__rewards_file: str = f"{base_experiment_path}src/main/resources/experiment_data/rewards_{rel_experiment_path}.csv"
        self.__positions_file: str = f"{base_experiment_path}src/main/resources/experiment_data/positions_{rel_experiment_path}.csv"
        self.__elapsed_times = []
        self.__rewards = []
        self.__coordinates = []
        self.__t_start = time.time()

    def save_data(self, avg_rewards, coordinates):
        print(avg_rewards, coordinates)
        self.__elapsed_times.append(time.time() - self.__t_start)
        self.save_rewards(avg_rewards)
        self.save_positions(coordinates)

    def save_rewards(self, avg_rewards):

        self.__rewards.append(avg_rewards)
        df_rewards = pd.DataFrame({"elapsed_time": self.__elapsed_times,
                                   "avg_rewards": self.__rewards})
        df_rewards.to_csv(self.__rewards_file)

    def save_positions(self, coordinates):
        positions_dict = {"elapsed_time": self.__elapsed_times}
        for i, coordinate in enumerate(coordinates):
            positions_dict.update({f"x_{i}": coordinate[0], f"y_{i}": coordinate[1]})
        df_positions = pd.DataFrame(positions_dict)
        df_positions.to_csv(self.__positions_file)
