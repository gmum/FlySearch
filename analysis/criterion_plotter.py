import numpy as np
import scipy

from typing import List, Dict, Callable
from analysis.run import Run
from analysis.run_analyser import RunAnalyser


class CriterionPlotter:
    def __init__(self, runs: List[Run]):
        self.runs = runs

    def aggregate_runs_per_function(self, fun: Callable, fil: Callable | None = None):
        class_to_run = {}

        if fil is None:
            fil = lambda x: True

        for run in self.runs:
            if not fil(run):
                continue

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

        variable_to_conf_ints = {
            variable: scipy.stats.binomtest(k=np.sum(successes), n=len(successes),
                                            alternative='two-sided').proportion_ci(confidence_level=0.95)
            for (variable, successes) in variable_to_successes.items()
        }

        conf_ints = np.array(
            [(variable_to_means[variable] - conf_int[0], conf_int[1] - variable_to_means[variable]) for
             variable, conf_int in variable_to_conf_ints.items()])

        conf_ints = np.transpose(conf_ints)

        print(conf_ints)

        plot = ax.bar(variable_to_means.keys(), variable_to_means.values(), yerr=conf_ints, capsize=5)

        for idx, rect in enumerate(plot):
            ax.text(rect.get_x() + rect.get_width() / 2.0, 0,
                    f"n={len(list(variable_to_successes.values())[idx])}",
                    ha='center', va='bottom')
