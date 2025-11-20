import logging
import pathlib
from datetime import datetime
from enum import Enum
from typing import Optional

import typer
from dotenv import load_dotenv

from analysis.results import print_results
from conversation.conversations import LLM_BACKEND_FACTORIES, LLMBackends
from prompts.prompts import PROMPT_FACTORIES
from rl.agents.agents import AGENT_FACTORIES, Agents
from rl.environment import EnvironmentType
from rl.environment.environments import ENVIRONMENTS
from rl.evaluation.configs.difficulty_levels import (
    DIFFICULTY_LEVELS,
    DifficultySettings,
)
from rl.evaluation.configs.experiment_config import ExperimentConfig
from rl.evaluation.experiment_runner import ExperimentRunner
from rl.evaluation.loggers.local_fs_logger_factory import LocalFSLoggerFactory
from rl.evaluation.validators.altitude_validator_factory import AltitudeValidatorFactory
from rl.evaluation.validators.out_of_bounds_flight_validator_factory import (
    OutOfBoundsFlightValidatorFactory,
)
from rl.evaluation.validators.reckless_flying_validator_factory import (
    RecklessFlyingValidatorFactory,
)
from scenarios.mimic_scenario_mapper import MimicScenarioMapper
from scenarios.scenarios import SCENARIO_CLASSES, Scenarios

app = typer.Typer(
    help="FlySearch benchmark", add_completion=False, no_args_is_help=True
)

load_dotenv(verbose=True)


class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


context = {}


@app.callback()
def main(
    model_backend: LLMBackends = typer.Option(help="The backend of the model to use"),
    model_name: str = typer.Option(
        help="The name of the model to use (passed to the model backend)"
    ),
    run_name: Optional[str] = typer.Option(
        help="The name of the benchmark run (default to date and time)", default=None
    ),
    results_directory: pathlib.Path = typer.Option(
        help="The directory to store the experiment results", default="all_logs"
    ),
    agent: Agents = typer.Option(
        help="The type of agent to use (use default for oryginal FlySearch)",
        default=Agents.SIMPLE_LLM,
    ),
    skip_sanity_check: bool = typer.Option(
        False,
        "--skip-sanity-check",
        help="Whether to skip running a sanity check before the benchmark (not recommended)",
    ),
    number_of_runs: int = typer.Option(
        help="The number of runs to perform",
        default=300,
    ),
    continue_from_idx: int = typer.Option(
        help="The index of the scenario to continue running from (e.g. if execution was interrupted)",
        default=0,
    ),
    log_level: LogLevel = typer.Option(
        help="The level of logging to use",
        default=LogLevel.INFO,
    ),
):
    logging.basicConfig(level=getattr(logging, log_level.value))

    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    context["conversation_factory"] = LLM_BACKEND_FACTORIES[model_backend](model_name)
    context["log_directory"] = pathlib.Path(results_directory) / run_name
    context["logger_factories"] = [
        LocalFSLoggerFactory(
            context["log_directory"],
            initial_iteration=continue_from_idx,
        )
    ]
    context["agent_factory"] = AGENT_FACTORIES[agent](context["conversation_factory"])
    context["continue_from_idx"] = continue_from_idx
    context["sanity_check"] = not skip_sanity_check
    context["number_of_runs"] = number_of_runs


@app.command()
def benchmark(
    scenario_directory: pathlib.Path = typer.Argument(
        help="The directory containing the scenarios to run the benchmark on"
    ),
):
    """
    Run a predefined benchmark set.
    """
    scenario_mapper = MimicScenarioMapper(scenario_directory)
    env_type = (
        EnvironmentType.CITY if scenario_mapper.is_city else EnvironmentType.FOREST
    )
    difficulty_level = DIFFICULTY_LEVELS[scenario_mapper.difficulty]
    environment = ENVIRONMENTS[env_type](
        give_class_image=difficulty_level.show_visual_sample,
        skip_object_placement_checks=True,
        require_object_in_sight=False,
    )

    validator_factories = [
        AltitudeValidatorFactory(difficulty_level.max_uav_altitude),
        OutOfBoundsFlightValidatorFactory(
            fs2_behavior=difficulty_level == DifficultySettings.FS_2
        ),
    ]
    if difficulty_level == DifficultySettings.FS_1:
        validator_factories.append(RecklessFlyingValidatorFactory())

    config = ExperimentConfig(
        agent_factory=context["agent_factory"],
        scenario_mapper=scenario_mapper,
        environment=environment,
        logger_factories=context["logger_factories"],
        validator_factories=validator_factories,
        forgiveness=difficulty_level.max_retries,
        number_of_runs=context["number_of_runs"],
        continue_from_idx=context["continue_from_idx"],
        number_of_glimpses=difficulty_level.max_steps,
        prompt_factory=PROMPT_FACTORIES[difficulty_level.prompt_type],
    )

    runner = ExperimentRunner(config, first_dummy=context["sanity_check"])
    runner.run()
    print_results(context["log_directory"])


@app.command()
def random_scenarios(
    scenario_type: Scenarios = typer.Argument(help="The type of scenario to generate"),
    difficulty: DifficultySettings = typer.Argument(
        help="The difficulty of the scenario"
    ),
):
    """
    Run FlySearch with random scenario generation.
    """
    difficulty_level = DIFFICULTY_LEVELS[difficulty]
    kwargs = {}
    # TODO: make better abstraction for this on refactor.
    if difficulty == DifficultySettings.FS_2:
        kwargs["random_sun"] = True
    scenario_mapper = SCENARIO_CLASSES[scenario_type](
        drone_alt_min=difficulty_level.starting_uav_altitude_range[0],
        drone_alt_max=difficulty_level.starting_uav_altitude_range[1],
        alpha=difficulty_level.starting_uav_position_offset,
        **kwargs,
    )
    environment = ENVIRONMENTS[
        EnvironmentType.CITY
        if scenario_type in [Scenarios.CITY, Scenarios.CITY_ANOMALY]
        else EnvironmentType.FOREST
    ](
        give_class_image=difficulty_level.show_visual_sample,
        require_object_in_sight=difficulty_level.target_line_of_sight_assured,
    )

    validator_factories = [
        AltitudeValidatorFactory(difficulty_level.max_uav_altitude),
        OutOfBoundsFlightValidatorFactory(
            fs2_behavior=difficulty_level == DifficultySettings.FS_2
        ),
    ]
    if difficulty_level == DifficultySettings.FS_1:
        validator_factories.append(RecklessFlyingValidatorFactory())

    config = ExperimentConfig(
        agent_factory=context["agent_factory"],
        scenario_mapper=scenario_mapper,
        environment=environment,
        logger_factories=context["logger_factories"],
        validator_factories=validator_factories,
        forgiveness=difficulty_level.max_retries,
        number_of_runs=context["number_of_runs"],
        continue_from_idx=context["continue_from_idx"],
        number_of_glimpses=difficulty_level.max_steps,
        prompt_factory=PROMPT_FACTORIES[difficulty_level.prompt_type],
    )

    runner = ExperimentRunner(config, first_dummy=context["sanity_check"])
    runner.run()
    print_results(context["log_directory"])


if __name__ == "__main__":
    app()
