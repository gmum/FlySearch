import pathlib

from rl.evaluation.loggers import BaseLoggerFactory
from rl.evaluation.loggers.local_fs_logger import LocalFSLogger


class LocalFSLoggerFactory(BaseLoggerFactory):
    def __init__(self, log_dir_prefix: pathlib.Path):
        self.log_dir = log_dir_prefix
        self.iteration = 0

    def get_logger(self):
        logger = LocalFSLogger(self.log_dir / str(self.iteration))
        self.iteration += 1

        return logger
