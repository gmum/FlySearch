import pathlib

from rl.evaluation.loggers import BaseLoggerFactory
from rl.evaluation.loggers.local_fs_logger import LocalFSLogger


class LocalFSLoggerFactory(BaseLoggerFactory):
    def __init__(self, log_dir_prefix: pathlib.Path, initial_iteration: int = 0):
        self.log_dir = log_dir_prefix
        self.iteration = initial_iteration

    def get_logger(self):
        logger = LocalFSLogger(self.log_dir / str(self.iteration))
        self.iteration += 1

        return logger
