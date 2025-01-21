import pathlib
import json

from matplotlib import pyplot as plt

from analysis import Run, RunAnalyser, CriterionPlotter, load_all_runs_from_a_dir


def main():
    path = pathlib.Path("../all_logs/MC-0S-F-10G-GPT-CR")
    runs = load_all_runs_from_a_dir(path)
    plotter = CriterionPlotter(runs)

    # Can be used for forest as well
    def city_aggregation_function(run: Run):
        object_type = str(run.object_type).lower()

        if "car" in object_type:
            return "car"
        elif "pickup" in object_type:
            return "car"
        elif "truck" in object_type:
            return "car"
        return object_type

    def success_criterion(run):
        return run.model_claimed and RunAnalyser(run).maciek_criterion_satisfied(10)

    runs_aggregated_per_type = plotter.aggregate_runs_per_function(city_aggregation_function)

    fig, ax = plt.subplots(nrows=1)
    stats = plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax, success_criterion=success_criterion)

    pretty_stats = json.dumps(stats, indent=4)

    print(pretty_stats)


if __name__ == "__main__":
    main()
