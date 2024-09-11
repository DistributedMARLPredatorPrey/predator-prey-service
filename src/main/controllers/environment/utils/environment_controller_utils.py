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
            elapsed_times = [float(et) for et in df_coordinates["elapsed_time"]]
            num_agents = len(
                [column for column in df_coordinates.columns if column.startswith("x")]
            )

            rewards = []
            for index, row in df_rewards.iterrows():
                row_rewards = []
                for i in range(num_agents):
                    row_rewards.append(row[f"r_{i}"])
                rewards.append(row_rewards)

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
            os.path.join(common_path, f"rewards1_{rel_experiment_path}.csv"),
            os.path.join(common_path, f"positions1_{rel_experiment_path}.csv"),
        )

    def save_data(self, rewards, coordinates):
        self.__elapsed_times.append(time.time() - self.__t_start)
        self.save_rewards(rewards)
        self.save_coordinates(coordinates)

    def save_rewards(self, rewards):
        if len(rewards) > 0:
            self.__rewards.append(rewards)

            rewards_dict = {"elapsed_time": self.__elapsed_times}
            for i in range(len(self.__rewards[0])):
                curr_rewards = [reward[i] for reward in self.__rewards]
                rewards_dict.update({f"r_{i}": curr_rewards})

            df_rewards = pd.DataFrame(rewards_dict)
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
