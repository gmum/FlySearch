from typing import List, Tuple

from PIL import Image, ImageDraw

from conversation import BaseConversationFactory, GPTFactory
from conversation.gemini_factory import GeminiFactory
from rl.agents.semantic_units import SemanticUnit
from rl.agents.semantic_units.detection_specialist import PivotLikeDetector, VLMVerifier, GoalIdentifier, \
    AggregatedPivotLikeDetector


class SimpleDetectionSpecialist(SemanticUnit):
    def __init__(self, conversation_factory: BaseConversationFactory, aggregation_level: int = 3):
        super().__init__(subunit_list=[
            AggregatedPivotLikeDetector(conversation_factory=conversation_factory, iterations=3,
                                        retries=aggregation_level),
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

    conversation_factory = GeminiFactory("gemini-2.0-flash")
    detection_specialist = SimpleDetectionSpecialist(conversation_factory=conversation_factory)
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/all_logs/GPT4o-CityNew/292_r0 /0.png")

    target = "an anomaly"
    annotated_image, detections = detection_specialist.get_detections(image, target)

    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
