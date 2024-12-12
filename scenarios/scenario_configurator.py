from time import sleep

from scenarios.classes_to_ids import classes_to_ids
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator


class ScenarioConfigurator:
    def __init__(self, glimpse_generator: UnrealGlimpseGenerator):
        self.glimpse_generator = glimpse_generator

    def hide_all_movable_objects(self):
        for key, value in classes_to_ids.items():
            if key != "SUN" and key != "FOREST":
                self.glimpse_generator.client.request(f"vset /object/{value}/hide")

    def show_object(self, object_name):
        object_id = classes_to_ids[object_name]
        self.glimpse_generator.client.request(f"vset /object/{object_id}/show")

    def move_object(self, object_name, x, y, z):
        object_id = classes_to_ids[object_name]
        self.glimpse_generator.client.request(f"vset /object/{object_id}/location {x} {y} {z}")

    def configure_scenario(self, scenario_dict):
        if "object_coords" in scenario_dict:
            self.glimpse_generator.change_start_position(scenario_dict["object_coords"])
            self.glimpse_generator.reset_camera()

        if "set_object" in scenario_dict and scenario_dict["set_object"]:
            object_name = scenario_dict["object_type"]
            object_coords = scenario_dict["object_coords"]

            self.hide_all_movable_objects()
            self.show_object(object_name)
            self.move_object(object_name, *object_coords)

        if "sun_y" in scenario_dict and "sun_z" in scenario_dict:
            sun_y = scenario_dict["sun_y"]
            sun_z = scenario_dict["sun_z"]
            sun_name = classes_to_ids["SUN"]

            self.glimpse_generator.client.request(f"vset /{sun_name}/rotation 0 {sun_y} {sun_z}")

        if "regenerate_forest" in scenario_dict and scenario_dict["regenerate_forest"]:
            forest_live_trees_density = scenario_dict["forest_live_trees_density"]
            forest_dead_trees_density = scenario_dict["forest_dead_trees_density"]
            forest_stones = scenario_dict["forest_stones"]
            forest_cliffs = scenario_dict["forest_cliffs"]
            seed = scenario_dict["seed"]

            forest_generator_name = classes_to_ids["FOREST"]

            self.glimpse_generator.client.request(f'vbp {forest_generator_name} RunPCG {forest_live_trees_density} {forest_dead_trees_density} {forest_stones} {forest_cliffs} {seed}')

            ready = self.glimpse_generator.client.request(f'vbp {forest_generator_name} IsPCGReady')

            while "true" not in ready :
                print("PCG is not ready, sleeping for 0.5 seconds, got:", ready)
                exit()
                sleep(0.5)