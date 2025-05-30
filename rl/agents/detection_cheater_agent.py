from typing import Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw

from rl.agents import DescriptionLLMAgent


class DetectionCheaterAgent(DescriptionLLMAgent):
    """
    DO NOT USE THIS AGENT FOR ANYTHING ELSE THAN ABLATION TESTING. It uses "cheat" field in the observation to obtain perfect detections of the object. Used to test performance of the agent conditioned that it's able to detect the object perfectly. DO NOT USE IN FS2, it assumes that we are in FS1 and the agent will be seen at the start.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_cheats = None
        self.annotations = []

    def _act(self, image: np.ndarray, altitude: np.ndarray, collision: int, **kwargs):
        self.current_cheats = kwargs["cheats"]
        return super()._act(image, altitude, collision, **kwargs)

    def _get_description_and_detection(self, image: Image.Image) -> Tuple[str, bool, str, Image.Image]:
        assert self.current_cheats is not None, "Cheats are not set. You need to call _act() first, which happens during normal operation of the agent."

        drone_position = list(map(int, list(self.current_cheats["real_position"])))
        x, y, altitude = drone_position

        if abs(x) > altitude or abs(y) > altitude:
            self.annotations.append(image)
            return "You had one job: fly to the object in red and you somehow managed to fail.", False, "Revert to your previous position and rethink your life choices.", image

        # Object is at -x, -y because the coordinate system is centered on it.
        x = -x
        y = -y

        axis_viewrange = 2 * altitude
        axis_pixels = 500

        pixels_per_meter = axis_pixels / axis_viewrange

        x_pixel = x * pixels_per_meter + axis_pixels // 2
        y_pixel = y * pixels_per_meter + axis_pixels // 2

        x_low = int(x_pixel - 50)
        x_high = int(x_pixel + 50)
        y_low = int(y_pixel - 50)
        y_high = int(y_pixel + 50)

        draw = ImageDraw.Draw(image)

        draw.rectangle([x_low, y_low, x_high, y_high], outline="red", width=2)

        self.annotations.append(image)

        return "Object is outlined in red", True, "Object is outlined in red", image

    def get_agent_info(self) -> Dict:
        return {
            "conversation_history": self.conversation.get_conversation(),
            "annotations": self.annotations,
        }
