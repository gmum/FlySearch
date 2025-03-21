from typing import Dict

from rl.evaluation import EvaluationState


class BaseLogger:
    def log(self, evaluation_state: EvaluationState):
        raise NotImplementedError

    def log_termination(self, termination_info: Dict):
        raise NotImplementedError
