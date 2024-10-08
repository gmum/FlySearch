import numpy as np

from PIL import Image

class AbstractOpenDetector:
    def __init__(self, threshold: float, image: np.ndarray):
        self.threshold = threshold
        self.image = image

    def detect(self, object_name_list: str) -> tuple[Image, list[tuple[float, float, float, float]], list[float]]:
        pass

    def get_image(self) -> np.ndarray:
        return self.image