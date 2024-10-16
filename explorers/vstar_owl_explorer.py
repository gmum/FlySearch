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
    def get_prompt(self, question, options, detector_worked = True): return f"""
        You will have to answer a question about the image. The object in question may be very small or almost impossible to see. {'To counter this, you will be given a version of the image with (potential) objects of interest highlighted in red, as well as cropped versions of them. Note that this technique is imperfect and there may be false positives or negatives.' if detector_worked else ''} Your task is to answer the question based on the information provided. Choose on of the options specified; to choose it, just copy its contents. Don't write anything else. ALWAYS PICK AN OPTION, EVEN IF YOU'RE NOT FULLY CERTAIN. THERE IS ALWAYS A GOOD OPTION. IF YOU REALLY DON'T KNOW, JUST PICK THE MOST LIKELY OPTION.
        
        Example:
        Question: Is the dog on the left side of the red car?
        Options: ["Yes", "No"]
        Answer: Yes
        
        Now, it's your turn.
        
        Question: {question}
        Options: {options}
    """
    
    def __init__(self, image, conversation: Conversation, question: str, options: list[str]):
        self.image = image
        self.conversation = conversation
        self.question = question
        self.options = options

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
            base_detector=Owl2Detector(0.2, image)
        )

    def identify_objects(self) -> list[str]:

        object_identifier = self._get_new_object_identifier()

        return object_identifier.identify_key_objects(self.question)


    def detect_objects(self, objects: list[str]) -> tuple[Image, list[Image]]:
        return self._get_new_visual_detector(self.image).detect_many_objects(objects)

    def get_answer(self) -> tuple[str, Image, list[Image], list[str]]:
        objects = self.identify_objects()
        image, cut_outs = self.detect_objects(objects)

        detector_worked = (len(cut_outs) != 0)

        prompt = self.get_prompt(self.question, self.options, detector_worked=detector_worked)

        if not detector_worked:
            # Better quality from the unnormalised one
            image = opencv_to_pil(self.image)

        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message(prompt)
        self.conversation.add_image_message(image)

        for cut_out in cut_outs:
            self.conversation.add_image_message(cut_out)

        self.conversation.commit_transaction(send_to_vlm=True)

        response = self.conversation.get_latest_message()[1]

        return response, image, cut_outs, objects

def main():
    from datasets.vstar_bench_dataset import VstarSubBenchDataset
    from conversation.openai_conversation import OpenAIConversation
    from openai import OpenAI
    from misc.config import OPEN_AI_KEY

    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(
        client,
        model_name="gpt-4o",
    )

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/direct_attributes", transform=pil_to_opencv)

    image, question, options, answer = ds[1]

    print(question)
    print(options)
    print(answer)

    explorer = VstarOwlExplorer(image, conversation, question, options)
    response, _, _, _ = explorer.get_answer()

    print("Response:", response)

if __name__ == "__main__":
    main()