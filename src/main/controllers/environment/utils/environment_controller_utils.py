import os
import time

import numpy as np
import pandas as pd


class EnvironmentControllerUtils:
    def __init__(self, base_experiment_path, rel_experiment_path):
        self.__rewards_file, self.__coordinates_file = self.__experiment_files_path(
            base_experiment_path, rel_experiment_path
        )
        self.__elapsed_times = []
        self.__rewards = []
        self.__coordinates = []
        self.__t_start = time.time()

    @staticmethod
    def __experiment_files_path(base_experiment_path, rel_experiment_path):
        common_path = os.path.join(
            base_experiment_path, "src", "main", "resources", "experiment_data"
        )
        return (
            os.path.join(common_path, f"rewards_{rel_experiment_path}.csv"),
            os.path.join(common_path, f"positions_{rel_experiment_path}.csv"),
        )

    # def __init_fields(self):
    #     self.__elapsed_times, self.__rewards, self.__coordinates = [], [], []
    #     if os.path.exists(self.__rewards_file):
    #         df_rewards = pd.read_csv(self.__rewards_file)
    #         self.__elapsed_times = list(df_rewards["elapsed_time"])
    #         self.__rewards = list(df_rewards["avg_rewards"])
    #     if os.path.exists(self.__coordinates_file):
    #         df_coord = pd.read_csv(self.__rewards_file)
    #         self.__coordinates = list(df_coord[""])

    def save_data(self, avg_rewards, coordinates):
        print(avg_rewards, coordinates)
        self.__elapsed_times.append(time.time() - self.__t_start)
        self.save_rewards(avg_rewards)
        self.save_coordinates(coordinates)

    def save_rewards(self, avg_rewards):
        self.__rewards.append(avg_rewards)
        df_rewards = pd.DataFrame(
            {"elapsed_time": self.__elapsed_times, "avg_rewards": self.__rewards}
        )
        df_rewards.to_csv(self.__rewards_file)

    def save_coordinates(self, coordinates):
        positions_dict = {"elapsed_time": self.__elapsed_times}
        self.__coordinates.append(coordinates)
        for i in range(len(self.__coordinates[0])):
            curr_coords = [coord[i] for coord in self.__coordinates]
            positions_dict.update(
                {
                    f"x_{i}": [c[0] for c in curr_coords],
                    f"y_{i}": [c[1] for c in curr_coords],
                }
            )
        df_positions = pd.DataFrame(positions_dict)
        df_positions.to_csv(self.__coordinates_file)