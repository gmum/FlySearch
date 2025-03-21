import os
import pathlib

from conversation import GPTFactory
from glimpse_generators import UnrealGlimpseGenerator
from misc import opencv_to_pil
from rl.environment import ForestFlySearchEnv, CityFlySearchEnv
from rl.evaluation.configs import BasicConfig
from rl.evaluation.experiment_runner import ExperimentRunner
from scenarios import ForestScenarioMapper, DefaultForestScenarioMapper, DefaultForestAnomalyScenarioMapper, \
    MimicScenarioMapper
from scenarios.default_city_anomaly_scenario_mapper import DefaultCityAnomalyScenarioMapper


def main():
    os.environ[
        "FOREST_BINARY_PATH"] = "/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"

    os.environ[
        "FONT_LOCATION"] = "/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

    os.environ[
        "CITY_BINARY_PATH"] = "/home/dominik/MyStuff/simulator/CitySample/Binaries/Linux/CitySample"

    os.environ[
        "LOCATIONS_CITY_PATH"] = "/home/dominik/MyStuff/active-visual-gpt/locations_city.csv"

    def glimpse_gen_func(client):
        return UnrealGlimpseGenerator(client=client)

    conversation_factory = GPTFactory()
    environment = CityFlySearchEnv()
    # setattr(environment, "get_glimpse_generator", glimpse_gen_func)
    # scenario_mapper = DefaultCityAnomalyScenarioMapper(drone_alt_min=30, drone_alt_max=35)
    scenario_mapper = MimicScenarioMapper(
        path=pathlib.Path("run_templates/city-template"), filter_str="*", continue_from=1
    )

    # scenario = scenario_mapper.create_random_scenario(seed=3)
    # scenario["drone_rel_coords"] = (0, 0, 10)

    config = BasicConfig(
        conversation_factory=conversation_factory,
        environment=environment,
        scenario_mapper=scenario_mapper,
        number_of_runs=1,
        run_name="TESTREFACTOR22"
    )

    runner = ExperimentRunner(config, first_dummy=True)
    runner.run()

    exit()

    with environment:
        obs, _ = environment.reset(seed=None, options=scenario)
        print(obs)
        print(scenario)

        pil_image = opencv_to_pil(obs["image"])
        pil_image.show()


if __name__ == "__main__":
    main()
