from tkinter import Image

import cv2
import numpy as np

from misc.add_guardrails import dot_matrix_two_dimensional


# It's an abstract class.
class ImageGlimpseGenerator:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.image_size = image.shape[:2]

    def get_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> list[np.ndarray]:
        pass

    def get_entire_image(self) -> np.ndarray:
        return self.image

    def convert_proportional_coords_to_pixel(self, x1, y1, x2, y2):
        height = self.image_size[0]
        width = self.image_size[1]

        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        return x1, y1, x2, y2


class BasicImageGlimpseGenerator(ImageGlimpseGenerator):
    def get_raw_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        x1, y1, x2, y2 = self.convert_proportional_coords_to_pixel(x1, y1, x2, y2)

        return self.image[y1:y2, x1:x2, :]

    # Yes, one glimpse may constitute several images.
    def get_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> list[np.ndarray]:
        glimpse = self.get_raw_glimpse(x1, y1, x2, y2)

        return [glimpse]


class GridImageGlimpseGenerator(BasicImageGlimpseGenerator):
    def __init__(self, image: np.ndarray, splits: int):
        super().__init__(image)
        self.splits = splits

    def get_raw_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        x1, y1, x2, y2 = self.convert_proportional_coords_to_pixel(x1, y1, x2, y2)

        return dot_matrix_two_dimensional(self.image, self.splits, self.splits)[y1:y2, x1:x2, :]

    def get_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> list[np.ndarray]:
        glimpse = self.get_raw_glimpse(x1, y1, x2, y2)

        return [glimpse]


def main():
    from datasets.vstar_bench_dataset import VstarSubBenchDataset
    from misc.cv2_and_numpy import pil_to_opencv

    ds = VstarSubBenchDataset("/home/anonymous/vstar_bench/relative_position", transform=pil_to_opencv)
    img, _, _, _ = ds[0]

    glimpse_generator = GridImageGlimpseGenerator(img, 5, 5)
    glimpse = glimpse_generator.get_glimpse(0.0, 0.0, 1.0, 1.0)

    cv2.imshow("Glimpse", glimpse[0])
    cv2.waitKey(0)

    glimpse = glimpse_generator.get_glimpse(0.0, 0.0, 0.5, 0.5)
    cv2.imshow("Glimpse", glimpse[0])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
