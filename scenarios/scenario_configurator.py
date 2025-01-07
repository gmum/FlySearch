from time import sleep

from scenarios.classes_to_ids import get_classes_to_object_classes
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator
from scenarios.object_classes import BaseObjectClass, ForestEnvPCGClass, ForestSunClass, PCGClass


class ScenarioConfigurator:
    def __init__(self, glimpse_generator: UnrealGlimpseGenerator):
        self.glimpse_generator = glimpse_generator
        self.classes_to_ids = get_classes_to_object_classes(glimpse_generator.client)

    def get_bbox(self, object_id: str) -> str:
        bbox = self.glimpse_generator.client.request(f"vget /object/{object_id}/bounds")
        return bbox

    def hide_all_movable_objects(self) -> None:
        for object_class in self.classes_to_ids.values():
            if isinstance(object_class, BaseObjectClass):
                object_class.hide_all_objects()

    # Sets the camera in a given location and asks for camera image, ensuring that the map is loaded
    def load_map(self, x, y, z, drone_rel_x_semi, drone_rel_y_semi, drone_rel_z_semi) -> None:
        # Asking glimpse generator for a glimpse will effectively load the map in a given location
        self.glimpse_generator.change_start_position((x, y, z))
        self.glimpse_generator.reset_camera()
        self.glimpse_generator.get_camera_image((drone_rel_x_semi, drone_rel_y_semi, drone_rel_z_semi), force_move=True)

    def rel_to_real(self, x, y, z, x_rel, y_rel, z_rel):
        x_rel *= 100
        y_rel *= 100
        z_rel *= 100

        return x + x_rel, y + y_rel, z + z_rel

    # If during configuration the scenario configurator made another assumption about the scenario, it should be noted
    # in the scenario_dict. E.g. the object id should be noted in the scenario_dict, as its value is randomly sampled from a given class.
    def configure_scenario(self, scenario_dict, recovery_generator=None):
        setup_is_correct = False

        while not setup_is_correct:
            setup_is_correct = True
            if "object_coords" in scenario_dict:
                self.load_map(*scenario_dict["object_coords"], *scenario_dict["drone_rel_coords"])

            if "set_object" in scenario_dict and scenario_dict["set_object"]:
                seed = scenario_dict["seed"]

                object_type: BaseObjectClass = scenario_dict["object_type"]
                object_class = self.classes_to_ids[object_type]

                self.hide_all_movable_objects()

                object_id = object_class.move_and_show(*scenario_dict["object_coords"], seed)
                scenario_dict["object_id"] = object_id

                if "object_rot" in scenario_dict:
                    object_class.rotate_object(object_id, *scenario_dict["object_rot"])

                # Always keep that if last here.
                if "regenerate_city" in scenario_dict and scenario_dict["regenerate_city"]:
                    city_generator_class: PCGClass = self.classes_to_ids["CITY"]
                    city_generator_class.move_and_show(*scenario_dict["object_coords"], seed)

                drone_real_coords = self.rel_to_real(*scenario_dict["object_coords"],
                                                     *scenario_dict["drone_rel_coords"])

                can_see = object_class.can_be_seen_from(object_id, *drone_real_coords)
                setup_is_correct = can_see

            if not setup_is_correct and recovery_generator is None:
                raise ValueError(
                    "Could not configure scenario properly. Provide the recovery_generator to help recover from the error.")

            if not setup_is_correct and recovery_generator is not None:
                print("ScenarioConfigurator: Could not configure scenario properly. Trying to recover...")
                new_scenario = recovery_generator.create_random_scenario()
                del new_scenario["seed"]
                scenario_dict.update(new_scenario)

        if "sun_y" in scenario_dict and "sun_z" in scenario_dict:
            sun_y = scenario_dict["sun_y"]
            sun_z = scenario_dict["sun_z"]
            sun_class: ForestSunClass = self.classes_to_ids["SUN"]
            sun_class.set_sun_rotation(sun_y, sun_z)

        if "regenerate_forest" in scenario_dict and scenario_dict["regenerate_forest"]:
            forest_live_trees_density = scenario_dict["forest_live_trees_density"]
            forest_dead_trees_density = scenario_dict["forest_dead_trees_density"]
            forest_stones = scenario_dict["forest_stones"]
            forest_cliffs = scenario_dict["forest_cliffs"]
            seed = scenario_dict["seed"]

            forest_env_class: ForestEnvPCGClass = self.classes_to_ids["FOREST"]
            forest_env_class.run_pcg(seed, forest_live_trees_density, forest_dead_trees_density, forest_stones,
                                     forest_cliffs)

        sleep(1)  # Wait for the fire to start burning -- in case it's needed
