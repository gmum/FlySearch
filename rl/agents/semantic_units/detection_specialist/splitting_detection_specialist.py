from typing import Tuple, List

from PIL import Image, ImageDraw

from conversation import BaseConversationFactory, GPTFactory
from conversation.gemini_factory import GeminiFactory
from rl.agents.semantic_units.detection_specialist import SimpleDetectionSpecialist


class SplittingDetectionSpecialist(SimpleDetectionSpecialist):
    def __init__(self, conversation_factory: BaseConversationFactory, parts_per_axis: int = 1,
                 aggregation_level: int = 3):
        super().__init__(conversation_factory=conversation_factory, aggregation_level=aggregation_level)
        self.parts_per_axis = parts_per_axis

    def get_detections(self, image: Image.Image, target: str) -> Tuple[
        Image.Image, List[Tuple[int, int, int, int]]]:

        # Split image into smaller sections
        width, height = image.size
        part_width = width // self.parts_per_axis
        part_height = height // self.parts_per_axis

        all_detections = []

        for i in range(self.parts_per_axis):
            for j in range(self.parts_per_axis):
                left = i * part_width
                upper = j * part_height
                right = left + part_width
                lower = upper + part_height

                part_image = image.crop((left, upper, right, lower))

                img, detections = super().get_detections(part_image, target)

                def detection_coorection(det):
                    x1, y1, x2, y2 = det
                    return (x1 + left, y1 + upper, x2 + left, y2 + upper)

                detections = [detection_coorection(det) for det in detections]
                all_detections.extend(detections)

        _, full_detections = super().get_detections(image, target)
        all_detections.extend(full_detections)

        annotated_image = image.copy()

        for x1, y1, x2, y2 in all_detections:
            draw = ImageDraw.Draw(annotated_image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        return annotated_image, all_detections


def main():
    from matplotlib import pyplot as plt

    conversation_factory = GeminiFactory("gemini-2.0-flash")
    detection_specialist = SplittingDetectionSpecialist(conversation_factory=conversation_factory, parts_per_axis=2,
                                                        aggregation_level=3)
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/all_logs/GPT4o-CityNew/292_r0 /0.png")

    target = "a tank"
    annotated_image, detections = detection_specialist.get_detections(image, target)

    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
