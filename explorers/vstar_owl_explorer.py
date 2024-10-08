import numpy as np
import openai
import cv2

from PIL import Image
from openai import OpenAI

from misc.cv2_and_numpy import opencv_to_pil, pil_to_opencv
from explorers.system_1.key_object_identifier import KeyObjectIdentifier
from conversation.abstract_conversation import Conversation, Role
from conversation.openai_conversation import OpenAIConversation
from misc.config import OPEN_AI_KEY
from open_detection.owl_2_detector import Owl2Detector
from open_detection.general_open_visual_detector import GeneralOpenVisualDetector

class VstarOwlExplorer:
    def __init__(self, image, conversation: Conversation, question: str):
        self.image = image
        self.conversation = conversation
        self.question = question

    def _get_new_conversation(self):
        oai_client = OpenAI(api_key=OPEN_AI_KEY)
        conversation = OpenAIConversation(
            oai_client,
            model_name="gpt-4o",
        )

        return conversation

    def _get_new_object_identifier(self):
        conversation = self._get_new_conversation()

        return KeyObjectIdentifier(conversation)

    def _get_new_visual_detector(self, image: np.ndarray):
        return GeneralOpenVisualDetector(
            threshold=0.2,
            base_detector=Owl2Detector(0.2, image)
        )

    def identify_objects(self) -> list[str]:

        object_identifier = self._get_new_object_identifier()

        return object_identifier.identify_key_objects(self.question)


    def detect_objects(self, objects: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        image = self.image

        all_cut_outs = []

        # FIXME: This is very bad. Detector should support detection of multiple objects at once. Otherwise, it's not even funny.
        for object in objects:
            visual_detector = self._get_new_visual_detector(image)
            image, cut_outs = visual_detector.detect(object)

            image: Image
            cut_outs: list[Image]

            all_cut_outs.extend(cut_outs)

            image = pil_to_opencv(image)


        all_cut_outs = [pil_to_opencv(cut_out) for cut_out in all_cut_outs]
        return image, all_cut_outs

    def answer(self):
        objects = self.identify_objects()
        image, cut_outs = self.detect_objects(objects)

        cv2.imshow("Image", image)
        cv2.waitKey(0)

        for cut_out in cut_outs:
            cv2.imshow("Cut out", cut_out)
            cv2.waitKey(0)

def main():
    from datasets.vstar_bench_dataset import VstarSubBenchDataset

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/direct_attributes", transform=pil_to_opencv)

    image, question, options, answer = ds[1]

    explorer = VstarOwlExplorer(image, None, question)
    explorer.answer()

if __name__ == "__main__":
    main()