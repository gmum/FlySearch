import random
import pandas as pd
import os

import math

from enum import Enum
from typing import Tuple


class CityScenarioMapper:
    class ObjectType(Enum):
        ANOMALY = 0
        POLICE_CAR = 1
        BEIGE_SPORT_CAR = 2
        BLUE_SPORT_CAR = 3
        RED_SPORT_CAR = 4
        WHITE_SPORT_CAR = 5
        CONSTRUCTION_WORKS = 6
        FIRE = 7
        BLACK_PICKUP_TRUCK = 8
        GREEN_PICKUP_TRUCK = 9
        RED_PICKUP_TRUCK = 10
        WHITE_PICKUP_TRUCK = 11
        CROWD = 12
        TRASH = 13
        BLACK_TRUCK = 14
        WHITE_TRUCK = 15

    @staticmethod
    def sample_value_between(val_min, val_max):
        return random.uniform(val_min, val_max)

    @staticmethod
    def sample_object_from_object_probs(object_probs: dict[ObjectType | Tuple, float]):
        random_value = random.random()
        cumulative_probability = 0

        for obj_type, probability in object_probs.items():
            cumulative_probability += probability

            if random_value <= cumulative_probability:
                if type(obj_type) is tuple:
                    return random.choice(obj_type)

                return obj_type

    @staticmethod
    def sample_drone_position(object_x, object_y, drone_z):
        if drone_z < 0:
            raise ValueError("Drone height must be positive")

        drone_x_min = object_x - 0.5 * drone_z
        drone_x_max = object_x + 0.5 * drone_z

        drone_y_min = object_y - 0.5 * drone_z
        drone_y_max = object_y + 0.5 * drone_z

        drone_x = CityScenarioMapper.sample_value_between(drone_x_min, drone_x_max)
        drone_y = CityScenarioMapper.sample_value_between(drone_y_min, drone_y_max)

        return drone_x, drone_y

    def __init__(self, object_probs: dict[ObjectType | Tuple, float], drone_z_rel_min: float, drone_z_rel_max: float,
                 scenarios_number: int, seed_min: int, seed_max: int, x_min: float = -math.inf, x_max: float = math.inf,
                 y_min: float = -math.inf,
                 y_max: float = math.inf):
        self.object_probs = object_probs

        self.drone_z_rel_min = drone_z_rel_min
        self.drone_z_rel_max = drone_z_rel_max
        self.scenarios_number = scenarios_number

        self.seed_min = seed_min
        self.seed_max = seed_max

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

    def _validate_object_probs(self):
        total = sum(self.object_probs.values())

        # FIXME
        if total != 1:
            raise ValueError(f"Total probability must be 1, but got {total}")

        for obj_type in self.object_probs:
            if obj_type not in CityScenarioMapper.ObjectType and type(obj_type) is not tuple:
                raise ValueError(f"Invalid object type: {obj_type}")

            if type(obj_type) is tuple:
                for subclass in obj_type:
                    if subclass not in CityScenarioMapper.ObjectType:
                        raise ValueError(f"Invalid object type: {subclass}")

    def create_random_scenario(self):
        object_type = CityScenarioMapper.sample_object_from_object_probs(self.object_probs)
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

        seed = int(self.sample_value_between(self.seed_min, self.seed_max))

        return {
            "object_coords": (object_x, object_y, object_z),
            "object_rot": (object_rot_p, object_rot_q, object_rot_r),
            "object_type": object_type,
            "drone_rel_coords": (drone_x, drone_y, drone_z),
            "set_object": True,
            "regenerate_city": True,
            "seed": seed
        }

    def iterate_scenarios(self):
        seeds = random.sample(range(self.seed_min, self.seed_max), self.scenarios_number)

        for seed in seeds:
            scenario = self.create_random_scenario()
            scenario["seed"] = seed
            yield scenario


def main():
    csm = CityScenarioMapper(
        object_probs={
            (CityScenarioMapper.ObjectType.POLICE_CAR, CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR): 1.0,
        },
        drone_z_rel_min=0,
        drone_z_rel_max=10000,
        scenarios_number=10,
        seed_max=1000,
        seed_min=0,
    )

    for params in csm.iterate_scenarios():
        print(params)


if __name__ == "__main__":
    main()
