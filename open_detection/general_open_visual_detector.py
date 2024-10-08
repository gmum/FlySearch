import numpy as np
import torch
import torchvision
import cv2

from PIL import Image

from open_detection.abstract_open_detector import AbstractOpenDetector
from open_detection.abstract_open_visual_detector import AbstractOpenVisualDetector
from misc.cv2_and_numpy import pil_to_opencv, opencv_to_pil


class GeneralOpenVisualDetector(AbstractOpenVisualDetector):
    def __init__(self, threshold: float, image: np.ndarray, base_detector: AbstractOpenDetector):
        self.threshold = threshold
        self.image = image
        self.base_detector = base_detector

    def detect(self, object_name: str) -> Image:
        padded_image, boxes, _ = self.base_detector.detect(object_name)

        padded_image = pil_to_opencv(padded_image)
        padded_image = torch.tensor(padded_image).permute(2, 0, 1)

        boxes = torch.tensor(boxes)

        image = torchvision.utils.draw_bounding_boxes(
            image=padded_image,
            boxes=boxes,
            width=6
        )

        image = image.permute(1, 2, 0).numpy()
        image = opencv_to_pil(image)

        return image


def main():
    from open_detection.owl_2_detector import Owl2Detector
    image = cv2.imread("../data/sample_images/burger.jpeg")

    detector = GeneralOpenVisualDetector(0.2, image, Owl2Detector(0.2, image))
    image = detector.detect("burger")

    image.show()

if __name__ == "__main__":
    main()