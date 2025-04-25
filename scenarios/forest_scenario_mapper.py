import random

from enum import Enum
from typing import Tuple, Dict, Any

from scenarios import BaseScenarioMapper


class ForestScenarioMapper(BaseScenarioMapper):
    class ObjectType(Enum):
        FOREST_FIRE = 0
        CAMPSITE = 1
        TRASH_PILE = 2
        BUILDING = 3
        PERSON = 4
        ANOMALY = 5

    def __init__(self, object_probs: dict[ObjectType | Tuple, float], x_min: float, x_max: float, y_min: float,
                 y_max: float, z_min: float, z_max: float, drone_z_rel_min: float, drone_z_rel_max: float,
                 alpha: float = 0.5):
        super().__init__(object_probs, ForestScenarioMapper.ObjectType)

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.drone_z_rel_min = drone_z_rel_min
        self.drone_z_rel_max = drone_z_rel_max
        self.alpha = alpha

        self._validate_object_probs()

    def create_random_scenario(self, seed: int) -> Dict[str, Any]:
        object_x = self.sample_value_between(self.x_min, self.x_max)
        object_y = self.sample_value_between(self.y_min, self.y_max)
        object_z = self.sample_value_between(self.z_min, self.z_max)

        object_type = self.sample_object_from_object_probs()

        drone_z = self.sample_value_between(object_z + self.drone_z_rel_min, object_z + self.drone_z_rel_max)

        drone_x, drone_y = ForestScenarioMapper.sample_drone_position(object_x, object_y, drone_z, alpha=self.alpha)

        drone_x = drone_x - object_x
        drone_y = drone_y - object_y

        drone_x = int(drone_x / 100)
        drone_y = int(drone_y / 100)
        drone_z = int(drone_z / 100)

        sun_y = self.sample_value_between(-90, -10)
        sun_z = self.sample_value_between(0, 360)

        forest_trees_density = self.sample_value_between(0.01, 0.3)
        forest_stones = self.sample_value_between(0.0, 0.1)

        return {
            "object_coords": (object_x, object_y, object_z),
            "object_type": object_type,
            "drone_rel_coords": (drone_x, drone_y, drone_z),
            "seed": seed,
            "forest_live_trees_density": forest_trees_density * 0.9,
            "forest_dead_trees_density": forest_trees_density * 0.1,
            "forest_stones": forest_stones,
            "forest_cliffs": 0,
            "sun_y": sun_y,
            "sun_z": sun_z,
            "regenerate_forest": True,
            "set_object": True
        }

    def get_description(self, object_type):
        if object_type != ForestScenarioMapper.ObjectType.ANOMALY:
            return f"a {super().get_description(object_type)}"
        else:
            return "an object that doesn't fit in with the rest of the environment (an anomaly)"


def main():
    fsm = ForestScenarioMapper(
        x_min=-25600,
        x_max=102400,
        y_min=-25600,
        y_max=102400,
        z_min=0,
        z_max=1,
        drone_z_rel_min=3000,
        drone_z_rel_max=10000,
        object_probs={
            (ForestScenarioMapper.ObjectType.FOREST_FIRE, ForestScenarioMapper.ObjectType.CAMPSITE): 1.0,
        }
    )

    print(fsm.create_random_scenario(3))


if __name__ == "__main__":
    main()
