from typing import List, Callable

from rl.agents import BaseAgentFactory
from rl.environment import BaseFlySearchEnv
from rl.evaluation.loggers import BaseLogger
from rl.evaluation.validators import BaseValidator
from scenarios import BaseScenarioMapper


class ExperimentConfig:
    def __init__(self, agent_factory: BaseAgentFactory, environment: BaseFlySearchEnv,
                 scenario_mapper: BaseScenarioMapper, loggers: List[BaseLogger], validators: List[BaseValidator],
                 forgiveness: int, number_of_runs: int, number_of_glimpses: int,
                 prompt_factory: Callable[[int, str, int], str]):
        self.agent_factory = agent_factory
        self.environment = environment
        self.scenario_mapper = scenario_mapper
        self.loggers = loggers
        self.validators = validators
        self.forgiveness = forgiveness
        self.number_of_runs = number_of_runs
        self.number_of_glimpses = number_of_glimpses
        self.prompt_factory = prompt_factory
