import random

from rl.environment import BaseFlySearchEnv
from rl.evaluation import TrajectoryEvaluator
from rl.evaluation.configs import ExperimentConfig


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _run_single_experiment(self, seed: int, environment: BaseFlySearchEnv):
        trajectory_evaluator = TrajectoryEvaluator(
            agent_factory=self.config.agent_factory,
            environment=environment,
            max_glimpses=self.config.number_of_glimpses,
            scenario_mapper=self.config.scenario_mapper,
            loggers=self.config.loggers,
            validators=self.config.validators,
            seed=seed,
            forgiveness=self.config.forgiveness,
            prompt_factory=self.config.prompt_factory
        )

        trajectory_evaluator.evaluate()

    def run(self):
        seeds = random.sample(range(int(1e9)), self.config.number_of_runs)

        with self.config.environment as environment:
            for seed in seeds:
                self._run_single_experiment(seed, environment)
