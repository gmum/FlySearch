import numpy as np

from typing import Dict, Tuple
from PIL import Image

from conversation import Conversation, Role, BaseConversationFactory
from misc import opencv_to_pil
from navigators import AbstractDroneNavigator
from rl.agents import BaseAgent, SimpleLLMAgent


class DescriptionLLMAgent(SimpleLLMAgent):
    def __init__(self, conversation: Conversation, description_conversation_factory: BaseConversationFactory,
                 prompt: str,
                 navigator: AbstractDroneNavigator, object_desc: str):
        super().__init__(conversation, prompt, navigator)

        self.description_conversation_factory = description_conversation_factory
        self.object_desc = object_desc

    def _get_description_and_detection(self, image: Image.Image) -> Tuple[str, bool, str]:
        description_conversation = self.description_conversation_factory.get_conversation()
        description_conversation.begin_transaction(Role.USER)

        description_conversation.add_text_message(
            "Your task is meant to describe this image for a pilot flying a UAV. Please, describe OBJECTS you see in the image in the utmost detail. Provide description for everything you see with inclusion of the coordinates on the image.")
        description_conversation.add_image_message(image)
        description_conversation.commit_transaction(send_to_vlm=True)
        _, description = description_conversation.get_latest_message()

        description_conversation.begin_transaction(Role.USER)
        description_conversation.add_text_message(
            f"Do you see an object which description is {self.object_desc} in the image? Answer with YES or NO."
        )
        description_conversation.commit_transaction(send_to_vlm=True)
        _, detection = description_conversation.get_latest_message()
        detection = "yes" in detection.strip().lower()

        description_conversation.begin_transaction(Role.USER)
        description_conversation.add_text_message(
            "Now, please provide us with additional information you would like the navigator to know about the image -- and, if applicable, the object you believe to be the object of interest. ")
        description_conversation.commit_transaction(send_to_vlm=True)

        _, additional_info = description_conversation.get_latest_message()

        return description, detection, additional_info

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

        description, detection, additional_info = self._get_description_and_detection(image)
        self.conversation.add_text_message(
            f"Our specialised agent described the image as follows: {description}. \n\n The agent did {'not ' if not detection else ''} detect the object of interest in the image. Its judgement may be wrong, so ultimately you need to decide on your own. Furthermore, the agent provided the following additional information: {additional_info}")

        self.conversation.add_image_message(image)
        self.conversation.commit_transaction(send_to_vlm=True)

        _, response = self.conversation.get_latest_message()

        return self._return_action_from_response(response)

    def get_agent_info(self) -> Dict:
        return {
            "conversation_history": self.conversation.get_conversation()
        }
