import numpy as np

from typing import Dict

from conversation import Conversation, Role
from misc import opencv_to_pil
from navigators import AbstractDroneNavigator
from rl.agents import BaseAgent


class SimpleLLMAgent(BaseAgent):
    def __init__(self, conversation: Conversation, prompt: str, navigator: AbstractDroneNavigator):
        self.conversation = conversation
        self.prompt = prompt
        self.navigator = navigator
        self.uninitialised = True

    def _return_action_from_response(self, response: str) -> Dict:
        try:
            coordinate_change = self.navigator.get_diffs(response)
        except Exception:
            if self.navigator.get_claim(response):
                return {
                    "found": 1
                }
            else:
                raise  # Model's syntax error

        return {
            "coordinate_change": coordinate_change
        }

    def _act(self, image: np.ndarray, altitude: np.ndarray, collision: int, **_):
        image = opencv_to_pil(image)
        collision = True if collision == 1 else False
        self.conversation.begin_transaction(Role.USER)

        if self.uninitialised:
            self.conversation.add_text_message(self.prompt)
            self.uninitialised = False

        if collision:
            self.conversation.add_text_message(
                "Emergency stop; you've flown too close to something and would have hit it.")

        self.conversation.add_text_message(f"Your current altitude is {altitude.item()} meters above ground level.")
        self.conversation.add_image_message(image)
        self.conversation.commit_transaction(send_to_vlm=True)

        _, response = self.conversation.get_latest_message()

        return self._return_action_from_response(response)

    def act(self, observation: Dict) -> Dict:
        return self._act(**observation)

    def correct_previous_action(self, fail_reason: Dict):
        if self.uninitialised:
            raise ValueError("No action to correct")

        match fail_reason["reason"]:
            case "too_high":
                alt_before = fail_reason["alt_before"]
                alt_after = fail_reason["alt_after"]
                alt_max = fail_reason["alt_max"]

                self.conversation.begin_transaction(Role.USER)
                self.conversation.add_text_message(
                    f"This command would cause you to fly too high. You can't fly higher than {alt_max} meters. Your current altitude is {alt_before} meters, which means that you can only fly {alt_max - alt_before} meters higher."
                )

                self.conversation.commit_transaction(send_to_vlm=True)
            case "out_of_bounds":
                self.conversation.begin_transaction(Role.USER)

                xy_bound = fail_reason["xy_bound"]

                self.conversation.add_text_message(
                    f"This command would cause you to fly out of the search area's bounds. You can't fly further than {xy_bound} meters from the starting point in any axis."
                )

                self.conversation.commit_transaction(send_to_vlm=True)
            case "reckless":
                self.conversation.begin_transaction(Role.USER)
                self.conversation.add_text_message(
                    "This command would endanger the drone, as you would fly out of bounds of the last seen image, possibly flying into unknown territories, recklessly. Please adjust your command so that you don't fly out of bounds of the previous glimpse."
                )

                self.conversation.commit_transaction(send_to_vlm=True)
            case _:
                raise ValueError("Unknown fail reason")

        _, response = self.conversation.get_latest_message()

        return self._return_action_from_response(response)
