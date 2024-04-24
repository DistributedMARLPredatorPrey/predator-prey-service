import os

import yaml

from src.main.model.config.config import (
    EnvironmentConfig,
    ReplayBufferServiceConfig,
    LearnerServiceConfig,
)


class ConfigUtils:
    @staticmethod
    def _load_config(file_path):
        """
        Loads a config file.
        :param file_path: file's path
        :return: file's content
        """
        with open(file_path, "r") as conf:
            return yaml.safe_load(conf)

    def environment_configuration(self) -> EnvironmentConfig:
        """
        Creates an EnvironmentConfig object by extracting information from the config file,
        whose path is specified by GLOBAL_CONFIG_PATH environment variable.
        :return: environment config
        """
        env_conf = self._load_config(os.environ.get("GLOBAL_CONFIG_PATH"))[
            "environment"
        ]
        return EnvironmentConfig(
            x_dim=env_conf["x_dim"],
            y_dim=env_conf["y_dim"],
            num_predators=env_conf["num_predators"],
            num_preys=env_conf["num_preys"],
            acc_lower_bound=env_conf["acc_lower_bound"],
            acc_upper_bound=env_conf["acc_upper_bound"],
            num_states=env_conf["num_states"],
            num_actions=env_conf["num_actions"],
            r=env_conf["r"],
            vd=env_conf["vd"],
            life=env_conf["life"],
        )

    def replay_buffer_configuration(self) -> ReplayBufferServiceConfig:
        """
        Creates an ReplayBufferConfig object by extracting information from env variables
        :return: replay buffer config
        """
        return ReplayBufferServiceConfig(
            replay_buffer_host=os.environ.get("REPLAY_BUFFER_HOST"),
            replay_buffer_port=int(os.environ.get("REPLAY_BUFFER_PORT")),
        )

    def learner_service_configuration(self) -> LearnerServiceConfig:
        return LearnerServiceConfig(pubsub_broker=os.environ.get("BROKER_HOST"))
