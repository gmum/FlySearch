from typing import Dict, Tuple

from rl.evaluation import EvaluationState


class BaseValidator:
    """
    Base class for evaluation validators. These are used to check if agent's action makes sense. If it doesn't, the trajectory evaluator may prompt the agent to redo the action or terminate the evaluation.
    """

    def validate(self, evaluation_state: EvaluationState) -> Tuple[bool, Dict]:
        """
        :return: Tuple containing a boolean whether the action is valid. If it's not, the second dictionary should not be empty and should contain information about the invalid action.
        """
        raise NotImplementedError()

    def inform_about_starting_altitude(self, starting_altitude: int):
        # If validator doesn't need to know about starting altitude, it can ignore this method. It should be called before the first action is sampled (if needed).
        pass

    def nuke(self):
        raise NotImplementedError()