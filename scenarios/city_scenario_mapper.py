import random
import pandas as pd
import os
import math

from enum import Enum
from typing import Tuple, Any, Dict

from scenarios import BaseScenarioMapper


class CityScenarioMapper(BaseScenarioMapper):
    class ObjectType(Enum):
        ANOMALY = 0
        POLICE_CAR = 1
        BEIGE_SPORT_CAR = 2
        BLUE_SPORT_CAR = 3
        RED_SPORT_CAR = 4
        WHITE_SPORT_CAR = 5
        ROAD_CONSTRUCTION_SITE = 6
        FIRE = 7
        BLACK_PICKUP_TRUCK = 8
        GREEN_PICKUP_TRUCK = 9
        RED_PICKUP_TRUCK = 10
        WHITE_PICKUP_TRUCK = 11
        CROWD = 12
        LARGE_TRASH_PILE = 13
        BLACK_TRUCK = 14
        WHITE_TRUCK = 15

    def __init__(self, object_probs: dict[ObjectType | Tuple, float], drone_z_rel_min: float, drone_z_rel_max: float,
                 x_min: float = -math.inf, x_max: float = math.inf,
                 y_min: float = -math.inf,
                 y_max: float = math.inf):
        super().__init__(object_probs, CityScenarioMapper.ObjectType)

        self.drone_z_rel_min = drone_z_rel_min
        self.drone_z_rel_max = drone_z_rel_max

        possible_location_csv_path = os.getenv("LOCATIONS_CITY_PATH")

        if possible_location_csv_path is None:
            raise ValueError(
                "LOCATIONS_CITY_PATH environment variable must be set to the path of the locations_city.csv file")

        self.possible_locations = pd.read_csv(possible_location_csv_path)

        # Drop locations that are not in the specified range
        self.possible_locations = self.possible_locations[self.possible_locations["X"] >= x_min]
        self.possible_locations = self.possible_locations[self.possible_locations["X"] <= x_max]

        self.possible_locations = self.possible_locations[self.possible_locations["Y"] >= y_min]
        self.possible_locations = self.possible_locations[self.possible_locations["Y"] <= y_max]

        self._validate_object_probs()

    def create_random_scenario(self, seed: int) -> Dict[str, Any]:
        object_type = self.sample_object_from_object_probs()
        row = self.possible_locations.sample(n=1).iloc[0]

        object_x = row["X"]
        object_y = row["Y"]
        object_z = row["Z"]

        # Rotations
        object_rot_p = row["P"]
        object_rot_q = row["Q"]
        object_rot_r = row["R"]

        drone_z = self.sample_value_between(object_z + self.drone_z_rel_min, object_z + self.drone_z_rel_max)

        drone_x, drone_y = CityScenarioMapper.sample_drone_position(object_x, object_y, drone_z)

        drone_x = drone_x - object_x
        drone_y = drone_y - object_y

        drone_x = int(drone_x / 100)
        drone_y = int(drone_y / 100)
        drone_z = int(drone_z / 100)

        return {
            "object_coords": (object_x, object_y, object_z),
            "object_rot": (object_rot_p, object_rot_q, object_rot_r),
            "object_type": object_type,
            "drone_rel_coords": (drone_x, drone_y, drone_z),
            "set_object": True,
            "regenerate_city": True,
            "seed": seed
        }

    def get_description(self, object_type):
        if object_type != CityScenarioMapper.ObjectType.ANOMALY:
            return f"a {super().get_description(object_type)}"
        else:
            return "an object that doesn't fit in with the rest of the environment (an anomaly)"


def main():
    csm = CityScenarioMapper(
        object_probs={
            (CityScenarioMapper.ObjectType.POLICE_CAR, CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR): 1.0,
        },
        drone_z_rel_min=0,
        drone_z_rel_max=10000,
    )

    print(csm.create_random_scenario(6))


if __name__ == "__main__":
    main()
