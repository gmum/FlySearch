import numpy as np

from typing import List, Dict, Callable
from analysis.run import Run
from analysis.run_analyser import RunAnalyser


class CriterionPlotter:
    def __init__(self, runs: List[Run]):
        self.runs = runs

    def aggregate_runs_per_function(self, fun: Callable):
        class_to_run = {}

        for run in self.runs:
            cls = fun(run)

            if cls not in class_to_run:
                class_to_run[cls] = []

            class_to_run[cls].append(run)

        return class_to_run

    def get_runs_aggregated_per_type(self) -> Dict:
        return self.aggregate_runs_per_function(lambda run: run.object_type)

    def get_runs_aggregated_per_height_bin(self) -> Dict:
        return self.aggregate_runs_per_function(lambda run: run.start_position[2] // 10)

    def plot_accuracy_in_aggregated_runs(self, variable_to_runs: Dict, ax, success_criterion: Callable | None = None,
                                         threshold=10):

        def run_was_successful(run):
            return run.model_claimed and RunAnalyser(run).maciek_criterion_satisfied(threshold)

        if success_criterion is None:
            success_criterion = run_was_successful

        variable_to_successes = {
            variable: np.array([success_criterion(run) for run in runs])
            for variable, runs in variable_to_runs.items()
        }

        variable_to_means = {
            variable: np.mean(successes)
            for variable, successes in variable_to_successes.items()
        }

        variable_to_stderr = {
            variable: np.std(successes) / (len(successes) ** 0.5)
            for variable, successes in variable_to_successes.items()
        }

        ax.bar(variable_to_means.keys(), variable_to_means.values(), yerr=variable_to_stderr.values(), capsize=5)
