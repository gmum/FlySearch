import random

from enum import Enum

class ForestScenarioMapper:
    class ObjectType(Enum):
        FIRE = 0
        TENT = 1
        TRASH = 2
        BUILDINGS = 3
        PEOPLE = 4
        ANOMALY = 5

    @staticmethod
    def sample_value_between(val_min, val_max):
        return random.uniform(val_min, val_max)

    @staticmethod
    def sample_object_from_object_probs(object_probs: dict[ObjectType, float]):
        random_value = random.random()
        cumulative_probability = 0

        for obj_type, probability in object_probs.items():
            cumulative_probability += probability

            if random_value <= cumulative_probability:
                return obj_type

    @staticmethod
    def sample_drone_position(object_x, object_y, drone_z):
        if drone_z < 0:
            raise ValueError("Drone height must be positive")

        # Triangle similarity for fov = 90 degrees
        drone_x_min = object_x - drone_z
        drone_x_max = object_x + drone_z

        drone_y_min = object_y - drone_z
        drone_y_max = object_y + drone_z

        drone_x = ForestScenarioMapper.sample_value_between(drone_x_min, drone_x_max)
        drone_y = ForestScenarioMapper.sample_value_between(drone_y_min, drone_y_max)

        return drone_x, drone_y

    def __init__(self, object_probs: dict[ObjectType, float], x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, drone_z_rel_min: float, drone_z_rel_max: float, seed_min: int, seed_max: int):
        self.object_probs = object_probs

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.drone_z_rel_min = drone_z_rel_min
        self.drone_z_rel_max = drone_z_rel_max
        self.seed_min = seed_min
        self.seed_max = seed_max

        self._validate_object_probs()

    def _validate_object_probs(self):
        total = sum(self.object_probs.values())

        if total != 1:
            raise ValueError(f"Total probability must be 1, but got {total}")

        for obj_type in self.object_probs:
            if obj_type not in ForestScenarioMapper.ObjectType:
                raise ValueError(f"Invalid object type: {obj_type}")

    def create_random_scenario(self):
        seed = int(self.sample_value_between(self.seed_min, self.seed_max))
        object_x = self.sample_value_between(self.x_min, self.x_max)
        object_y = self.sample_value_between(self.y_min, self.y_max)
        object_z = self.sample_value_between(self.z_min, self.z_max)

        object_type = ForestScenarioMapper.sample_object_from_object_probs(self.object_probs)

        drone_z = self.sample_value_between(object_z + self.drone_z_rel_min, object_z + self.drone_z_rel_max)

        drone_x, drone_y = ForestScenarioMapper.sample_drone_position(object_x, object_y, drone_z)
        
        return {
            "object_coords": (object_x, object_y, object_z),
            "object_type": object_type,
            "drone_coords":(drone_x, drone_y, drone_z),
            "seed": seed,
            "forest_density": 0.1,
            "forest_stones": 0.1,
            "forest_cliffs": 0.0
        }