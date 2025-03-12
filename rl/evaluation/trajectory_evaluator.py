from typing import Callable, List

from rl.evaluation.validators import BaseValidator
from rl.evaluation.loggers import BaseLogger
from rl.agents import BaseAgent
from rl.environment import BaseFlySearchEnv, DroneCannotSeeTargetException
from scenarios import BaseScenarioMapper


class TrajectoryEvaluator:
    """
    Class tasked with performing a single evaluation of the agent in the environment.
    """

    def __init__(self, agent: BaseAgent, environment: BaseFlySearchEnv, max_glimpses: int,
                 prompt_generator: Callable[[int, str, int], str], scenario_mapper: BaseScenarioMapper,
                 loggers: List[BaseLogger], validators: List[BaseValidator], seed: int):
        self.agent = agent
        self.environment = environment
        self.max_glimpses = max_glimpses
        self.prompt_generator = prompt_generator
        self.scenario_mapper = scenario_mapper
        self.loggers = loggers
        self.validators = validators
        self.seed = seed

        if not self.environment.resources_initialized:
            raise ValueError("Environment resources not initialized. Use `with` statement before using the evaluator.")

        self._prepare_environment()

    def _prepare_environment(self):
        throws = 0

        while True:
            scenario = self.scenario_mapper.create_random_scenario(seed=self.seed)
            try:
                self.environment.reset(seed=self.seed, options=scenario)
            except DroneCannotSeeTargetException:
                throws += 1
            else:
                break

        scenario["throws"] = throws

    def evaluate(self):
        pass
