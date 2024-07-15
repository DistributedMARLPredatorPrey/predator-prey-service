import os
import time

import pandas as pd


class EnvironmentControllerUtils:
    def __init__(self, base_experiment_path, rel_experiment_path):
        self.__rewards_file, self.__coordinates_file = self.__experiment_files_path(
            base_experiment_path, rel_experiment_path
        )
        self.__elapsed_times, self.__rewards, self.__coordinates = (
            self.__load_existing()
        )
        self.__t_start = time.time()

    def __load_existing(self):
        if os.path.exists(self.__rewards_file) and os.path.exists(
            self.__coordinates_file
        ):
            df_rewards = pd.read_csv(self.__rewards_file)
            df_coordinates = pd.read_csv(self.__coordinates_file)
            elapsed_times = [float(et) for et in df_rewards["elapsed_time"]]
            rewards = [float(r) for r in df_rewards["avg_rewards"]]
            num_agents = len(
                [column for column in df_coordinates.columns if column.startswith("x")]
            )
            coords = []
            for index, row in df_coordinates.iterrows():
                row_coords = []
                for i in range(num_agents):
                    row_coords.append((row[f"x_{i}"], row[f"y_{i}"]))
                coords.append(row_coords)
            return elapsed_times, rewards, coords
        return [], [], []

    @staticmethod
    def __experiment_files_path(base_experiment_path, rel_experiment_path):
        common_path = os.path.join(
            base_experiment_path, "src", "main", "resources", "experiment_data"
        )
        return (
            os.path.join(common_path, f"rewards_{rel_experiment_path}.csv"),
            os.path.join(common_path, f"positions_{rel_experiment_path}.csv"),
        )

    def save_data(self, avg_rewards, coordinates):
        self.__elapsed_times.append(time.time() - self.__t_start)
        self.save_rewards(avg_rewards)
        self.save_coordinates(coordinates)

    def save_rewards(self, avg_reward):
        self.__rewards.append(avg_reward)
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
