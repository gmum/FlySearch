import numpy as np

from PIL import Image
from typing import Tuple, Dict

from conversation import BaseConversationFactory, Role
from rl.agents import DescriptionLLMAgent
from rl.agents.semantic_units.decision_making_specialist import DecisionMakingSpecialist
from rl.agents.semantic_units.detection_specialist import SplittingDetectionSpecialist


class DetectionDrivenDescriptionLLMAgent(DescriptionLLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.annotations = []
        self.returns = []

    def _get_description_and_detection(self, image: Image.Image) -> Tuple[str, bool, str, Image.Image]:
        detector = SplittingDetectionSpecialist(self.description_conversation_factory, parts_per_axis=2,
                                                aggregation_level=1)

        annotated_image, detection_coords = detector.get_detections(image, self.object_desc)
        self.annotations.append(annotated_image)

        if len(detection_coords) == 0:
            to_return = super()._get_description_and_detection(image)
            self.returns.append(to_return[:3])
            return to_return

        conversation = self.description_conversation_factory.get_conversation()
        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            f"You are going to be presented with an image with POTENTIAL detections of {self.object_desc}. Those are provided by an independent AI detector that could be utterly wrong. As such, you should tell us WHICH of them make sense by providing their coordinates. Your output will be forwarded to another decision-maker that won't see these annotations, so you need to be as precise as possible. For now, please describe the image, objects you can find in it, and their locations. Ignore annotations that are pointless. Focus on things that look like {self.object_desc}.", )
        conversation.add_image_message(annotated_image)
        conversation.commit_transaction(send_to_vlm=True)

        _, description = conversation.get_latest_message()
        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            f"Now, please tell me if you found {self.object_desc} in the image. Respond ONLY with YES or NO.")
        conversation.commit_transaction(send_to_vlm=True)
        _, found = conversation.get_latest_message()
        found = "yes" in found.lower()

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            f"Now, please provide us with some additional information about the image for the decision-maker. ")
        conversation.commit_transaction(send_to_vlm=True)
        _, additional_info = conversation.get_latest_message()

        to_return = (description, found, additional_info, annotated_image)
        self.returns.append(
            to_return[:3])  # Store only the first three elements (we don't want to store the image that way)
        return to_return

    def get_agent_info(self) -> Dict:
        return {
            "conversation_history": self.conversation.get_conversation(),
            "annotations": self.annotations,
            "returns": self.returns
        }
