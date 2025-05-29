from typing import List, Tuple, Dict

from PIL import Image, ImageDraw

from conversation import BaseConversationFactory, Role, GPTFactory
from rl.agents.semantic_units.detection_specialist import BaseVerifier


class VLMVerifier(BaseVerifier):
    def __init__(self, conversation_factory: BaseConversationFactory):
        """
        Initialize the VLMVerifier with a conversation factory.
        :param conversation_factory: Factory to create conversations.
        """
        self.conversation_factory = conversation_factory

    def validate_detection(self, image: Image.Image, target: str, detection: Tuple[int, int, int, int]) -> bool:
        """
        Validate the detection using the conversation factory.
        :param image: Image to be validated.
        :param target: Target to be validated.
        :param detection: Detection to be validated.
        :return: True if the detection is valid, False otherwise.
        """
        # Create a conversation using the conversation factory
        conversation = self.conversation_factory.get_conversation()

        # Cut out the image using the detection coordinates
        x1, y1, x2, y2 = detection
        cutout = image.crop((x1, y1, x2, y2))

        # Check if the cutout is empty
        if cutout.getbbox() is None:
            return False

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            f"Does this image contain {target} or something that could resemble {target}? Please answer with 'yes' or 'no'. The object is cut out from the larger picture. The cutout is noted on the larger picture.")

        image_with_bbox = image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        conversation.add_image_message(image_with_bbox)
        conversation.add_image_message(cutout)
        conversation.commit_transaction(send_to_vlm=True)

        _, response = conversation.get_latest_message()

        return "yes" in response.lower()

    def process_information(self, information: Dict) -> Dict:
        """
        Process the information received from the main agent and return a dictionary with the processed information.
        :param information: Dictionary with the information to be processed.
        :return: Dictionary with the processed information.
        """
        image = information["image"]
        target = information["target"]
        detections = information["detections"]

        valid_detections = [
            detection for detection in detections if self.validate_detection(image, target, detection)
        ]

        information["detections"] = valid_detections

        return information


def main():
    conversation_factory = GPTFactory()
    verifier = VLMVerifier(conversation_factory=conversation_factory)

    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")
    target = "something funny"
    detection = (0, 0, 500, 500)  # Example detection coordinates

    is_valid = verifier.validate_detection(image, target, detection)

    print(f"Is the detection valid? {is_valid}")


if __name__ == "__main__":
    main()
