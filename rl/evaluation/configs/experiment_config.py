from typing import List, Callable

from rl.agents.base_agent_factory import BaseAgentFactory
from rl.environment.base_fly_search_env import BaseFlySearchEnv
from rl.evaluation.loggers.base_logger_factory import BaseLoggerFactory
from rl.evaluation.validators.base_validator_factory import BaseValidatorFactory
from scenarios.base_scenario_mapper import BaseScenarioMapper


class ExperimentConfig:
    def __init__(self, agent_factory: BaseAgentFactory, environment: BaseFlySearchEnv,
                 scenario_mapper: BaseScenarioMapper, logger_factories: List[BaseLoggerFactory],
                 validator_factories: List[BaseValidatorFactory],
                 forgiveness: int, number_of_runs: int, continue_from_idx: int, number_of_glimpses: int,
                 prompt_factory: Callable[[int, str, int], str]):
        self.agent_factory = agent_factory
        self.environment = environment
        self.scenario_mapper = scenario_mapper
        self.logger_factories = logger_factories
        self.validator_factories = validator_factories
        self.forgiveness = forgiveness
        self.number_of_runs = number_of_runs
        self.continue_from_idx = continue_from_idx
        self.number_of_glimpses = number_of_glimpses
        self.prompt_factory = prompt_factory
