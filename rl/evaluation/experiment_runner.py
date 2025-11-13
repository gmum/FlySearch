import random
from typing import Sequence

from rl.environment.base_fly_search_env import BaseFlySearchEnv
from rl.evaluation.configs.experiment_config import ExperimentConfig
from rl.evaluation.trajectory_evaluator import TrajectoryEvaluator


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, first_dummy: bool = True):
        self.config = config
        self.dummy = first_dummy

        self.number_of_runs = config.number_of_runs
        if isinstance(config.scenario_mapper, Sequence):
            self.number_of_runs = min(
                self.number_of_runs, len(config.scenario_mapper) - self.config.continue_from_idx
            )

    def _run_single_experiment(self, run_idx: int, seed: int, running_environment: BaseFlySearchEnv):
        if not self.dummy:
            for logger_factory in self.config.logger_factories:
                if logger_factory.exists(run_idx):
                    print(f"Skipping run {run_idx} as logs already exist")
                    return

        validators = [
            validator_factory.get_validator()
            for validator_factory in self.config.validator_factories
        ]

        self.config.scenario_mapper.set_seed(seed)

        trajectory_evaluator = TrajectoryEvaluator.prepare_simulator(
            agent_factory=self.config.agent_factory,
            environment=running_environment,
            max_glimpses=self.config.number_of_glimpses,
            scenario_mapper=self.config.scenario_mapper,
            validators=validators,
            seed=seed,
            forgiveness=self.config.forgiveness,
            prompt_factory=self.config.prompt_factory,
            scenario_idx=run_idx
        )
        if not trajectory_evaluator:
            print(f"Skipping run {run_idx} due to invalid scenario")
            return

        if self.dummy:
            print("Performing dummy run")
            loggers = []
            self.dummy = False
        else:
            print(f"Performing run with seed {seed}")
            loggers = [
                logger_factory.get_logger_for_scenario(run_idx)
                for logger_factory in self.config.logger_factories
            ]

        trajectory_evaluator.evaluate(loggers)

    def _run_experiments(self, running_environment: BaseFlySearchEnv):
        if self.dummy:
            self._run_single_experiment(self.config.continue_from_idx, 42,
                                        running_environment)  # Dummy run for sanity check

        seeds = random.sample(
            range(int(1e9)), self.config.number_of_runs
        )
        for idx, seed in enumerate(seeds):
            self._run_single_experiment(self.config.continue_from_idx + idx, seed, running_environment)

    def run(self):
        with self.config.environment as running_environment:
            self._run_experiments(running_environment)
