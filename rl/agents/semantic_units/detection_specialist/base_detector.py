from PIL import Image
from typing import Dict, List, Tuple

from rl.agents.semantic_units import SemanticUnit, BaseSemanticSubunit


class BaseDetector(BaseSemanticSubunit):
    def image_to_detections(self, image: Image.Image, target: str) -> List[Tuple[int, int, int, int]]:
        """
        Convert an image to a list of detections.
        :param image: Image to be converted.
        :param target: Target to be detected.
        :return: List of detections.
        """
        raise NotImplementedError

    def process_information(self, information: Dict) -> Dict:
        image = information["image"]
        target = information["target"]
        detections = self.image_to_detections(image, target)

        information["detections"] = detections
        return information
