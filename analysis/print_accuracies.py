import pathlib
import json

from matplotlib import pyplot as plt

from analysis import Run, RunAnalyser, CriterionPlotter, load_all_runs_from_a_dir

def main():
    path = pathlib.Path("../all_logs/Gemma27b-FS2")
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


    runs_aggregated_per_type = plotter.aggregate_runs_per_function(city_aggregation_function)

    fig, ax = plt.subplots(nrows=1)
    stats = plotter.plot_accuracy_in_aggregated_runs(
        runs_aggregated_per_type,
        ax,
        success_criterion=lambda run: RunAnalyser(run).success_criterion_satisfied(threshold=10, check_claimed=True)
    )

    print("CLAIMED STATS")
    print(json.dumps(stats, indent=4))

    runs_unaggregated = plotter.aggregate_runs_per_function(
        lambda x: "unaggregated" if "anomaly" not in str(x.object_type).lower() else "anomaly")

    stats = plotter.plot_accuracy_in_aggregated_runs(
        runs_unaggregated,
        ax,
        success_criterion=lambda run: RunAnalyser(run).success_criterion_satisfied(threshold=10, check_claimed=False)
    )

    print("UNAGGREGATED CLAIMED STATS")
    print(json.dumps(stats, indent=4))


    stats = plotter.plot_accuracy_in_aggregated_runs(
        runs_aggregated_per_type,
        ax,
        success_criterion=lambda x: RunAnalyser(x).success_criterion_satisfied(threshold=10, check_claimed=False)
    )

    print("UNCLAIMED STATS")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
