import numpy as np
import torch
import torchvision
import cv2

from PIL import Image

from open_detection.abstract_open_detector import AbstractOpenDetector
from open_detection.abstract_open_visual_detector import AbstractOpenVisualDetector
from misc.cv2_and_numpy import pil_to_opencv, opencv_to_pil


class GeneralOpenVisualDetector(AbstractOpenVisualDetector):
    def __init__(self, base_detector: AbstractOpenDetector):
        self.base_detector = base_detector

    def _cut_out_objects(self, padded_image: np.ndarray, boxes) -> list[Image]:
        subimages = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            subimage = padded_image[y1:y2, x1:x2]
            subimages.append(subimage)

        return subimages

    def widen_boxes(self, boxes, padding: int, image_size) -> list[tuple[float, float, float, float]]:
        widened_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(image_size[1], x2)
            y2 = min(image_size[0], y2)

            widened_boxes.append((x1, y1, x2, y2))

        return widened_boxes

    def detect(self, object_name: str) -> tuple[Image, list[Image]]:
        padded_image, boxes, _ = self.base_detector.detect(object_name)

        if len(boxes) == 0:
            return padded_image, []

        padded_image = pil_to_opencv(padded_image)
        boxes = self.widen_boxes(boxes, 20, padded_image.shape[:2])

        cut_outs = self._cut_out_objects(padded_image, boxes)
        padded_image = torch.tensor(padded_image).permute(2, 0, 1)

        boxes = torch.tensor(boxes)

        image = torchvision.utils.draw_bounding_boxes(
            image=padded_image,
            boxes=boxes,
            width=6
        )

        image = image.permute(1, 2, 0).numpy()
        image = opencv_to_pil(image)

        cut_outs = [opencv_to_pil(cut_out) for cut_out in cut_outs]

        return image, cut_outs


def main():
    from open_detection.owl_2_detector import Owl2Detector
    image = cv2.imread("../data/sample_images/burger.jpeg")

    detector = GeneralOpenVisualDetector(Owl2Detector(0.2, image))
    image, cut_outs = detector.detect("burger")
    image = pil_to_opencv(image)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

    for cut_out in cut_outs:
        cut_out = pil_to_opencv(cut_out)
        cv2.imshow("Cut out", cut_out)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()