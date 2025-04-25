import random
import pandas as pd
import os

import math

from enum import Enum
from typing import Tuple, Dict, Any


class BaseScenarioMapper:
    def __init__(self, object_probs, object_type_cls):
        self.object_probs = object_probs
        self.object_type_cls = object_type_cls

    @staticmethod
    def sample_value_between(val_min, val_max):
        return random.uniform(val_min, val_max)

    def sample_object_from_object_probs(self):
        random_value = random.random()
        cumulative_probability = 0

        for obj_type, probability in self.object_probs.items():
            cumulative_probability += probability

            if random_value <= cumulative_probability:
                if type(obj_type) is tuple:
                    return random.choice(obj_type)

                return obj_type

    @staticmethod
    def sample_drone_position(object_x, object_y, drone_z, alpha=0.5):
        if drone_z < 0:
            raise ValueError("Drone height must be positive")

        drone_x_min = object_x - alpha * drone_z
        drone_x_max = object_x + alpha * drone_z

        drone_y_min = object_y - alpha * drone_z
        drone_y_max = object_y + alpha * drone_z

        drone_x = BaseScenarioMapper.sample_value_between(drone_x_min, drone_x_max)
        drone_y = BaseScenarioMapper.sample_value_between(drone_y_min, drone_y_max)

        return drone_x, drone_y

    def _validate_object_probs(self):
        total = sum(self.object_probs.values())

        # This function assumes that probabilities are "sane" and floating point errors won't skew the sum of probabilities. If this becomes an issue, consider changing the implementation.
        if total != 1:
            raise ValueError(f"Total probability must be 1, but got {total}")

        for obj_type in self.object_probs:
            if obj_type not in self.object_type_cls and type(obj_type) is not tuple:
                raise ValueError(f"Invalid object type: {obj_type}")

            if type(obj_type) is tuple:
                for subclass in obj_type:
                    if subclass not in self.object_type_cls:
                        raise ValueError(f"Invalid object type: {subclass}")

    def create_random_scenario(self, seed: int) -> Dict[str, Any]:
        """
        Creates a random scenario. Should be implemented in the subclass.
        :param seed: Seed FOR THE ENVIRONMENT. Does NOT affect the values of the scenario.
        :return: Dictionary with scenario parameters that can be passed to the gymnasium environment.
        """
        raise NotImplementedError()

    def get_description(self, object_type):
        if object_type not in self.object_type_cls:
            raise ValueError(f"Invalid object type: {object_type}")

        return str(object_type.name).replace("_", " ").lower()

    def get_object_type_cls(self):
        return self.object_type_cls

    def empty(self):
        return False
