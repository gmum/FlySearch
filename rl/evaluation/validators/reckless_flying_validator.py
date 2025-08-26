import numpy as np

from rl.evaluation import EvaluationState
from rl.evaluation.validators import BaseValidator


class RecklessFlyingValidator(BaseValidator):
    def validate(self, evaluation_state: EvaluationState):
        # raise ValueError(str(list(evaluation_state.info.keys())))
        try:
            previous_position = evaluation_state.info["real_position"]
        except KeyError:
            raise ValueError(str(list(evaluation_state.info.keys())))
        coord_change = np.abs(np.array(evaluation_state.action["coordinate_change"]))

        altitude = previous_position[2]

        if np.max(coord_change[:2]) > altitude:
            return False, {"reason": "reckless"}

        return True, {}

    def nuke(self):
        pass