import os

from conversation import GPTFactory
from rl.environment import ForestFlySearchEnv
from rl.evaluation.configs import BasicConfig
from rl.evaluation.experiment_runner import ExperimentRunner
from scenarios import ForestScenarioMapper, DefaultForestScenarioMapper


def main():
    os.environ[
        "FOREST_BINARY_PATH"] = "/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"

    os.environ[
        "FONT_LOCATION"] = "/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

    os.environ[
        "CITY_BINARY_PATH"] = "/home/dominik/MyStuff/simulator/CitySample/Binaries/Linux/CitySample"

    conversation_factory = GPTFactory()
    environment = ForestFlySearchEnv()
    scenario_mapper = DefaultForestScenarioMapper(drone_alt_min=50, drone_alt_max=80)

    config = BasicConfig(
        conversation_factory=conversation_factory,
        environment=environment,
        scenario_mapper=scenario_mapper,
        number_of_runs=1
    )

    runner = ExperimentRunner(config)

    runner.run()


if __name__ == "__main__":
    main()
