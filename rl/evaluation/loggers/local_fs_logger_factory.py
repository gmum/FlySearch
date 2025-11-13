import pathlib

from rl.evaluation.loggers.base_logger_factory import BaseLoggerFactory
from rl.evaluation.loggers.local_fs_logger import LocalFSLogger


class LocalFSLoggerFactory(BaseLoggerFactory):
    def __init__(self, log_dir_prefix: pathlib.Path, initial_iteration: int = 0):
        self.log_dir = log_dir_prefix
        self.iteration = initial_iteration

    def get_logger(self):
        logger = LocalFSLogger(self.iteration, self.log_dir / str(self.iteration))
        self.iteration += 1

        return logger

    def get_logger_for_scenario(self, scenario_idx: int) -> LocalFSLogger:
        return LocalFSLogger(scenario_idx, self.log_dir / str(scenario_idx))

    def exists(self, scenario_idx: int) -> bool:
        path = self.log_dir / str(scenario_idx)
        return path.exists() and path.is_dir()
