import pathlib

from prompts import xml_found_prompt
from rl.agents import SimpleLLMAgentFactory
from rl.environment import BaseFlySearchEnv, CityFlySearchEnv
from rl.evaluation.configs import ExperimentConfig
from rl.evaluation.loggers import WandbLoggerFactory, LocalFSLogger, LocalFSLoggerFactory
from rl.evaluation.loggers.wandb_logger import WandbLogger
from rl.evaluation.validators import OutOfBoundsFlightValidatorFactory, AltitudeValidatorFactory
from rl.evaluation.validators.reckless_flying_validator import RecklessFlyingValidator
from rl.evaluation.validators.reckless_flying_validator_factory import RecklessFlyingValidatorFactory
from scenarios import CityScenarioMapper, BaseScenarioMapper


class BasicConfig(ExperimentConfig):
    def __init__(self, conversation_factory, environment: BaseFlySearchEnv, scenario_mapper: BaseScenarioMapper,
                 number_of_runs: int, run_name: str):
        super().__init__(
            agent_factory=SimpleLLMAgentFactory(conversation_factory),
            environment=environment,
            scenario_mapper=scenario_mapper,
            logger_factories=[WandbLoggerFactory(project_name="WTLN-RL-T"),
                              LocalFSLoggerFactory(log_dir_prefix=pathlib.Path(f"all_logs/{run_name}"))],
            validator_factories=[RecklessFlyingValidatorFactory(),
                                 OutOfBoundsFlightValidatorFactory(search_diameter=200),
                                 AltitudeValidatorFactory(max_altitude=120)],
            forgiveness=5,
            number_of_runs=number_of_runs,
            number_of_glimpses=10,
            prompt_factory=xml_found_prompt
        )
