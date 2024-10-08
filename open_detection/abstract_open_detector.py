import numpy as np

class AbstractOpenDetector:
    def __init__(self, threshold: float, image: np.ndarray):
        self.threshold = threshold
        self.image = image

    def detect(self, object_name_list: str) -> tuple[tuple[float, float, float, float], float]:
        pass

    def get_image(self) -> np.ndarray:
        return self.image