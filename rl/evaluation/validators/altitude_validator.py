from rl.evaluation import EvaluationState
from rl.evaluation.validators import BaseValidator


class AltitudeValidator(BaseValidator):
    def __init__(self, max_altitude: int):
        self.max_altitude = max_altitude

    def validate(self, evaluation_state: EvaluationState):
        altitude_before = evaluation_state.info["real_position"][2]
        altitude_diff = evaluation_state.action["coordinate_change"][2]

        if altitude_before + altitude_diff > self.max_altitude:
            return False, {"reason": "too_high", "alt_before": altitude_before,
                           "alt_after": altitude_before + altitude_diff, "alt_max": self.max_altitude}

        return True, {}
