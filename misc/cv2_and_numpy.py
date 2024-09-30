import numpy as np
from PIL import Image


def pil_to_opencv(image: Image.Image) -> np.ndarray:
    return np.array(image)[:, :, ::-1].copy()


def opencv_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image[:, :, ::-1].copy())
