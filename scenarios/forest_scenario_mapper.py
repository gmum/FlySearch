from enum import Enum
from typing import Tuple, Dict, Any

from scenarios.base_scenario_mapper import EpisodeIteratorMapper


class ForestScenarioMapper(EpisodeIteratorMapper):
    class ObjectType(Enum):
        FOREST_FIRE = 0
        CAMPSITE = 1
        TRASH_PILE = 2
        BUILDING = 3
        PERSON = 4
        ANOMALY = 5

    def __init__(
            self,
            object_probs: dict[ObjectType | Tuple, float],
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            z_min: float,
            z_max: float,
            drone_z_rel_min: float,
            drone_z_rel_max: float,
            alpha: float = 0.5,
    ):
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

    def __next__(self) -> Dict[str, Any]:
        object_x = self.sample_value_between(self.x_min, self.x_max)
        object_y = self.sample_value_between(self.y_min, self.y_max)
        object_z = self.sample_value_between(self.z_min, self.z_max)

        object_type = self.sample_object_from_object_probs()

        drone_z = self.sample_value_between(
            object_z + self.drone_z_rel_min, object_z + self.drone_z_rel_max
        )

        drone_x, drone_y = ForestScenarioMapper.sample_drone_position(
            object_x, object_y, drone_z, alpha=self.alpha
        )

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
            "seed": self.seed,
            "forest_live_trees_density": forest_trees_density * 0.9,
            "forest_dead_trees_density": forest_trees_density * 0.1,
            "forest_stones": forest_stones,
            "forest_cliffs": 0,
            "sun_y": sun_y,
            "sun_z": sun_z,
            "regenerate_forest": True,
            "set_object": True,
        }

    def get_description(self, object_type):
        if object_type != ForestScenarioMapper.ObjectType.ANOMALY:
            return f"a {super().get_description(object_type)}"
        else:
            return "an object that doesn't fit in with the rest of the environment (an anomaly)"
