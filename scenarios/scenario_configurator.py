import json
import random

from time import sleep

from scenarios import ForestScenarioMapper, CityScenarioMapper
from scenarios.classes_to_ids import classes_to_ids
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator


class ScenarioConfigurator:
    def __init__(self, glimpse_generator: UnrealGlimpseGenerator):
        self.glimpse_generator = glimpse_generator

    # Warning: This method randomly samples object's id if a type has many ids associated with it
    def get_object_id(self, object_type):
        object_id = classes_to_ids[object_type]

        if isinstance(object_id, list):
            object_id = random.sample(object_id, 1)[0]

        return object_id

    def hide_all_movable_objects(self):
        for key, value in classes_to_ids.items():
            if isinstance(value, list):
                for v in value:
                    self.glimpse_generator.client.request(f"vset /object/{v}/hide")
            elif key != "SUN" and key != "FOREST" and key != "CITY":
                self.glimpse_generator.client.request(f"vset /object/{value}/hide")

    def show_object(self, object_id: str):
        self.glimpse_generator.client.request(f"vset /object/{object_id}/show")

    def get_bbox(self, object_id: str) -> str:
        bbox = self.glimpse_generator.client.request(f"vget /object/{object_id}/bounds")
        return bbox

    def move_object(self, object_id: str, x, y, z):
        self.glimpse_generator.client.request(f"vset /object/{object_id}/location {x} {y} {z}")

    def rotate_object(self, object_id: str, p, y, r):
        self.glimpse_generator.client.request(f"vset /object/{object_id}/rotation {p} {y} {r}")

    def wait_for_pcg(self, object_id: str):
        ready = json.loads(self.glimpse_generator.client.request(f'vbp {object_id} IsPCGReady'))["ready"]

        while ready == "false":
            ready = json.loads(self.glimpse_generator.client.request(f'vbp {object_id} IsPCGReady'))["ready"]
            print("PCG is not ready, sleeping for 0.5 seconds, got:", ready)
            sleep(0.5)

    # Sets the camera in a given location and asks for camera image, ensuring that the map is loaded
    def load_map(self, x, y, z, drone_rel_x_semi, drone_rel_y_semi, drone_rel_z_semi) -> None:
        # Asking glimpse generator for a glimpse will effectively load the map in a given location
        self.glimpse_generator.change_start_position((x, y, z))
        self.glimpse_generator.reset_camera()
        self.glimpse_generator.get_camera_image((drone_rel_x_semi, drone_rel_y_semi, drone_rel_z_semi), force_move=True)

    # If during configuration the scenario configurator made another assumption about the scenario, it should be noted
    # in the scenario_dict. E.g. the object id should be noted in the scenario_dict, as its value is randomly sampled from a given class.
    def configure_scenario(self, scenario_dict):
        if "regenerate_city" in scenario_dict and scenario_dict["regenerate_city"]:
            city_generator_name = self.get_object_id("CITY")
            self.glimpse_generator.client.request(f"vbp {city_generator_name} clear")
            sleep(1)

        if "object_coords" in scenario_dict:
            self.load_map(*scenario_dict["object_coords"], *scenario_dict["drone_rel_coords"])

        if "set_object" in scenario_dict and scenario_dict["set_object"]:
            object_type = scenario_dict["object_type"]
            object_coords = scenario_dict["object_coords"]
            object_id = self.get_object_id(object_type)

            scenario_dict["object_id"] = object_id

            self.hide_all_movable_objects()
            self.show_object(object_id)
            self.move_object(object_id, *object_coords)

            if "object_rot" in scenario_dict:
                self.rotate_object(object_id, *scenario_dict["object_rot"])

            if object_type == ForestScenarioMapper.ObjectType.CAMPING:
                seed = scenario_dict["seed"]
                self.glimpse_generator.client.request(f"vbp {object_id} RunPCG {seed}")

                # self.wait_for_pcg(object_id)
                sleep(1)

            if object_type == CityScenarioMapper.ObjectType.CROWD:
                seed = scenario_dict["seed"]
                self.glimpse_generator.client.request(f"vbp {object_id} RunPCG {seed}")

                # self.wait_for_pcg(object_id)
                sleep(2)

            if object_type == CityScenarioMapper.ObjectType.TRASH:
                seed = scenario_dict["seed"]
                self.glimpse_generator.client.request(f"vbp {object_id} RunPCG {seed}")

                # self.wait_for_pcg(object_id)
                sleep(2)

        if "sun_y" in scenario_dict and "sun_z" in scenario_dict:
            sun_y = scenario_dict["sun_y"]
            sun_z = scenario_dict["sun_z"]
            sun_name = self.get_object_id("SUN")

            self.glimpse_generator.client.request(f"vset /{sun_name}/rotation 0 {sun_y} {sun_z}")

        if "regenerate_forest" in scenario_dict and scenario_dict["regenerate_forest"]:
            forest_live_trees_density = scenario_dict["forest_live_trees_density"]
            forest_dead_trees_density = scenario_dict["forest_dead_trees_density"]
            forest_stones = scenario_dict["forest_stones"]
            forest_cliffs = scenario_dict["forest_cliffs"]
            seed = scenario_dict["seed"]

            forest_generator_name = self.get_object_id("FOREST")

            self.glimpse_generator.client.request(
                f'vbp {forest_generator_name} RunPCG {forest_live_trees_density} {forest_dead_trees_density} {forest_stones} {forest_cliffs} {seed}')

            self.wait_for_pcg(forest_generator_name)

        if "regenerate_city" in scenario_dict and scenario_dict["regenerate_city"]:
            city_generator_name = self.get_object_id("CITY")
            self.glimpse_generator.client.request(f"vbp {city_generator_name} spawn")
            sleep(1)  # Don't wait for PCG here as in the city scenario you can't do that
