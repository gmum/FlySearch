import os

from time import sleep
from typing import Dict

from scenarios.object_classes import BaseObjectClass, PCGClass, ForestEnvPCGClass, ForestSunClass

from glimpse_generators import UnrealClientWrapper
from rl import BaseFlySearchEnv


class ForestFlySearchEnv(BaseFlySearchEnv):
    def __init__(self, resolution: int = 500, max_altitude: int = 120):
        super().__init__(resolution=resolution, max_altitude=max_altitude)

    def get_client(self) -> UnrealClientWrapper:
        forest_binary_path = os.getenv("FOREST_BINARY_PATH")

        if forest_binary_path is None:
            raise ValueError("FOREST_BINARY_PATH environment variable not set and required for forest scenario type.")

        return UnrealClientWrapper(host="localhost", port=9000, unreal_binary_path=forest_binary_path)

    def _configure(self, options: Dict) -> None:
        options["cansee_restarts"] = 0

        if "object_coords" in options:
            self.load_map(*options["object_coords"], *options["drone_rel_coords"])

        if "set_object" in options and options["set_object"]:
            seed = options["seed"]

            object_type: BaseObjectClass = options["object_type"]
            object_class = self.classes_to_ids[object_type]

            self.hide_all_movable_objects()

            object_id = object_class.move_and_show(*options["object_coords"], seed)
            options["object_id"] = object_id

            if "object_rot" in options:
                object_class.rotate_object(object_id, *options["object_rot"])

        if "sun_y" in options and "sun_z" in options:
            sun_y = options["sun_y"]
            sun_z = options["sun_z"]
            sun_class: ForestSunClass = self.classes_to_ids["SUN"]
            sun_class.set_sun_rotation(sun_y, sun_z)

        if "regenerate_forest" in options and options["regenerate_forest"]:
            forest_live_trees_density = options["forest_live_trees_density"]
            forest_dead_trees_density = options["forest_dead_trees_density"]
            forest_stones = options["forest_stones"]
            forest_cliffs = options["forest_cliffs"]
            seed = options["seed"]

            forest_env_class: ForestEnvPCGClass = self.classes_to_ids["FOREST"]
            forest_env_class.run_pcg(seed, forest_live_trees_density, forest_dead_trees_density, forest_stones,
                                     forest_cliffs)

        print(options["object_type"])
        if 'FIRE' in str(options["object_type"]):
            sleep(10)  # Wait for the fire to start burning -- in case it's needed
        else:
            sleep(5)  # wait for the objects to load
