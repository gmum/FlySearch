import random

from response_parsers import ParsingError
from rl.environment import BaseFlySearchEnv
from rl.evaluation import TrajectoryEvaluator
from rl.evaluation.configs import ExperimentConfig


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, first_dummy: bool = True):
        self.config = config
        self.dummy = first_dummy

    def _run_single_experiment(self, seed: int, environment: BaseFlySearchEnv):
        if self.dummy:
            print("Performing dummy run")
            loggers = []
            self.dummy = False
        else:
            print(f"Performing run with seed {seed}")
            loggers = [logger_factory.get_logger() for logger_factory in self.config.logger_factories]

        validators = [validator_factory.get_validator() for validator_factory in self.config.validator_factories]

        trajectory_evaluator = TrajectoryEvaluator(
            agent_factory=self.config.agent_factory,
            environment=environment,
            max_glimpses=self.config.number_of_glimpses,
            scenario_mapper=self.config.scenario_mapper,
            loggers=loggers,
            validators=validators,
            seed=seed,
            forgiveness=self.config.forgiveness,
            prompt_factory=self.config.prompt_factory
        )

        trajectory_evaluator.evaluate()

    def run(self):
        seeds = random.sample(range(int(1e9)),
                              self.config.number_of_runs + (1 if self.dummy else 0))

        with self.config.environment as environment:
            for seed in seeds:
                self._run_single_experiment(seed, environment)
