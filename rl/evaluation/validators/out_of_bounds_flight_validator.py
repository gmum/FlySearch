import numpy as np

from rl.evaluation import EvaluationState


class OutOfBoundsFlightValidator:
    def __init__(self, search_diameter: int = 200):
        self.search_diameter = search_diameter
        self.first_position = None

    def validate(self, evaluation_state: EvaluationState):
        if self.first_position is None:
            self.first_position = evaluation_state.info["real_position"]

        current_position = evaluation_state.info["real_position"]
        diff = evaluation_state.action["coordinate_change"]

        total_diff = np.array(current_position) - np.array(self.first_position) + np.array(diff)

        if np.max(np.abs(total_diff[:2])) > self.search_diameter:
            return False, {"reason": "out_of_bounds", "xy_bound": self.search_diameter}

        return True, {}
