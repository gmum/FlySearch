from typing import List, Tuple

from PIL import Image, ImageDraw

from conversation import BaseConversationFactory, GPTFactory
from rl.agents.semantic_units import SemanticUnit
from rl.agents.semantic_units.detection_specialist import PivotLikeDetector, VLMVerifier


class SimpleDetectionSpecialist(SemanticUnit):
    def __init__(self, conversation_factory: BaseConversationFactory):
        super().__init__(subunit_list=[
            PivotLikeDetector(conversation_factory=conversation_factory, iterations=3),
            VLMVerifier(conversation_factory=conversation_factory),
        ])

    def get_detections(self, image: Image.Image, target: str) -> Tuple[
        Image.Image, List[Tuple[int, int, int, int]]]:
        """
        Get detections from the image using the detection specialist.
        :param image: Image to be processed.
        :param target: Target to be detected.
        :return: List of detections.
        """
        information = {
            "image": image,
            "target": target,
        }
        information = self.process_information(information)

        detections = information["detections"]

        annotated_image = image.copy()

        for x1, y1, x2, y2 in detections:
            draw = ImageDraw.Draw(annotated_image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        return annotated_image, detections


def main():
    from matplotlib import pyplot as plt

    conversation_factory = GPTFactory()
    detection_specialist = SimpleDetectionSpecialist(conversation_factory=conversation_factory)
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")

    target = "something funny"
    annotated_image, detections = detection_specialist.get_detections(image, target)

    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
