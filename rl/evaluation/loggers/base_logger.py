from abc import abstractmethod, ABC
from typing import Dict

from rl.evaluation.evaluation_state import EvaluationState


class LogExistsException(Exception):
    pass


class BaseLogger(ABC):
    def __init__(self, scenario_idx: int):
        self.scenario_idx = scenario_idx

    @abstractmethod
    def log(self, evaluation_state: EvaluationState):
        raise NotImplementedError

    @abstractmethod
    def log_termination(self, termination_info: Dict):
        raise NotImplementedError

    @abstractmethod
    def nuke(self):
        raise NotImplementedError
