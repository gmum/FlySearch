import json
import random

from time import sleep

from scenarios import ForestScenarioMapper
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

            if key != "SUN" and key != "FOREST":
                self.glimpse_generator.client.request(f"vset /object/{value}/hide")

    def show_object(self, object_id):
        self.glimpse_generator.client.request(f"vset /object/{object_id}/show")

    def get_bbox(self, object_id) -> str:
        bbox = self.glimpse_generator.client.request(f"vget /object/{object_id}/bounds")
        return bbox

    def move_object(self, object_id, x, y, z):
        self.glimpse_generator.client.request(f"vset /object/{object_id}/location {x} {y} {z}")

    def rotate_object(self, object_id, p, y, r):
        self.glimpse_generator.client.request(f"vset /object/{object_id}/rotation {p} {y} {r}")

    def wait_for_pcg(self, object_id):
        ready = json.loads(self.glimpse_generator.client.request(f'vbp {object_id} IsPCGReady'))["ready"]

        while ready == "false":
            ready = json.loads(self.glimpse_generator.client.request(f'vbp {object_id} IsPCGReady'))["ready"]
            print("PCG is not ready, sleeping for 0.5 seconds, got:", ready)
            sleep(0.5)

    def configure_scenario(self, scenario_dict):
        if "object_coords" in scenario_dict:
            self.glimpse_generator.change_start_position(scenario_dict["object_coords"])
            self.glimpse_generator.reset_camera()

        if "set_object" in scenario_dict and scenario_dict["set_object"]:
            object_type = scenario_dict["object_type"]
            object_coords = scenario_dict["object_coords"]
            object_id = self.get_object_id(object_type)

            self.hide_all_movable_objects()
            self.show_object(object_type)
            self.move_object(object_type, *object_coords)

            if "object_rot" in scenario_dict:
                self.rotate_object(object_type, *scenario_dict["object_rot"])

            if object_type == ForestScenarioMapper.ObjectType.CAMPING:
                seed = scenario_dict["seed"]
                self.glimpse_generator.client.request(f"vbp {object_id} RunPCG {seed}")

                self.wait_for_pcg(object_id)

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
