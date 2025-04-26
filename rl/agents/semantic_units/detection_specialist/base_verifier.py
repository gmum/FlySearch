from typing import List

from rl.agents.semantic_units import BaseSemanticSubunit


class BaseVerifier(BaseSemanticSubunit):
    def verify_detections(self, detections: List[int, int, int, int]) -> List[int, int, int, int]:
        """
        Verify the detections and return a list of verified detections.
        :param detections: List of detections to be verified.
        :return: List of verified detections.
        """

        raise NotImplementedError
