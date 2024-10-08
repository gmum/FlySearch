import numpy as np

from PIL import Image


class AbstractOpenVisualDetector:
    def detect(self, object_name: str) -> tuple[Image, list[Image]]:
        pass
