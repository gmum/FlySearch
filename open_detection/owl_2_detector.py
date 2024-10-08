import torchvision
import torchvision.utils
import numpy as np
import torch
import cv2

from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from typing import Iterable

from misc.cv2_and_numpy import opencv_to_pil, pil_to_opencv
from open_detection.abstract_open_detector import AbstractOpenDetector

class Owl2Detector(AbstractOpenDetector):
    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    def __init__(self, threshold: float, image: np.ndarray):
        super().__init__(threshold, image)

    def _get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def _detection_result_iterator(self, boxes, scores) -> Iterable[tuple[tuple[float, float, float, float], float]]:
        for box, score in zip(boxes, scores):
            box = [round(i, 2) for i in box.tolist()]
            yield tuple(box), score.item()

    def detect(self, object_name: str) -> list[tuple[tuple[float, float, float, float], float]]:
        object_list_of_lists = [[object_name]]

        image = opencv_to_pil(self.image)
        inputs = self.processor(text=object_list_of_lists, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        unnormalized_image = self._get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, _ = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        return list(self._detection_result_iterator(boxes, scores))

    def get_image(self) -> np.ndarray:
        return self.image

def main():
    image = cv2.imread("../data/sample_images/burger.jpeg")
    detector = Owl2Detector(0.2, image)
    print(detector.detect("burger"))

if __name__ == "__main__":
    main()