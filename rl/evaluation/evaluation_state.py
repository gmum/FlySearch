import numpy as np

from typing import Dict
from PIL import Image


class EvaluationState:
    def __init__(self, observation: Dict, action: Dict, info: Dict, observation_number: int, correction_number: int,
                 agent_info):
        self.observation = observation
        self.action = action
        self.info = info
        self.observation_number = observation_number
        self.correction_number = correction_number
        self.agent_info = agent_info
