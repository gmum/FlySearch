import numpy as np

from rl.evaluation import EvaluationState


class OutOfBoundsFlightValidator:
    def __init__(self, search_diameter: int = 200, fs2_behavior: bool = False):
        self.search_diameter = search_diameter
        self.first_position = None
        self.fs2_behavior = fs2_behavior

    def validate(self, evaluation_state: EvaluationState):
        if self.first_position is None:
            self.first_position = evaluation_state.info["real_position"]

        current_position = evaluation_state.info["real_position"]
        diff = list(evaluation_state.action["coordinate_change"])

        diff[1] = -diff[1]
        # Invert the y-axis change. Yes, I know. The fact that environment silently inverts the y-axis was a design choice of me.
        # Either way, we need to do this because otherwise bugs will happen.

        total_diff = np.array(current_position) - np.array(self.first_position) + np.array(diff)

        if np.max(np.abs(total_diff[:2])) > self.search_diameter:
            return False, {"reason": "out_of_bounds", "xy_bound": self.search_diameter}

        return True, {}

    def inform_about_starting_altitude(self, starting_altitude: float):
        if self.fs2_behavior:
            self.search_diameter = starting_altitude


    def nuke(self):
        self.first_position = None