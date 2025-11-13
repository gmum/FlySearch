from abc import ABC, abstractmethod

from rl.evaluation.loggers.base_logger import BaseLogger


class BaseLoggerFactory(ABC):

    @abstractmethod
    def get_logger(self) -> BaseLogger:
        raise NotImplementedError()

    @abstractmethod
    def get_logger_for_scenario(self, scenario_idx: int) -> BaseLogger:
        raise NotImplementedError()

    @abstractmethod
    def exists(self, scenario_idx: int) -> bool:
        raise NotImplementedError()
