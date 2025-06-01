import pathlib
import json

from matplotlib import pyplot as plt

from analysis import Run, RunAnalyser, CriterionPlotter, load_all_runs_from_a_dir

CITY_CATEGORIES = ['road_construction_site', 'crowd', 'large_trash_pile', 'fire', 'car']
FOREST_CATEGORIES = ['campsite', 'trash_pile', 'person', 'forest_fire', 'building']
ANOMALY_CATEGORY = ['anomaly']


def main():
    path = pathlib.Path("../all_logs/Gemma3-27b-City-FS1")
    # path = pathlib.Path("../all_logs/Claude35-FS2-Reckless1")
    runs = load_all_runs_from_a_dir(path)
    plotter = CriterionPlotter(runs)

    if "-an-" in str(path).lower():
        categories = ANOMALY_CATEGORY
    elif "-city-" in str(path).lower() or "fs2" in str(path).lower():
        categories = CITY_CATEGORIES
    elif "-f-" in str(path).lower():
        categories = FOREST_CATEGORIES
    else:
        raise ValueError(f"Path {path} does not contain a valid scenario")

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
        return run.model_claimed and RunAnalyser(run).success_criterion_satisfied(10)

    runs_aggregated_per_type = plotter.aggregate_runs_per_function(city_aggregation_function)

    fig, ax = plt.subplots(nrows=1)
    stats = plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax, success_criterion=success_criterion)

    pretty_stats = json.dumps(stats, indent=4)

    print("CLAIMED STATS")
    print(pretty_stats)

    print(' & '.join(f'${stats[c]['mean'] * 100:.1f}\\%$' for c in categories))

    runs_unaggregated = plotter.aggregate_runs_per_function(
        lambda x: "unaggregated" if "anomaly" not in str(x.object_type).lower() else "anomaly")
    stats = plotter.plot_accuracy_in_aggregated_runs(runs_unaggregated, ax,
                                                     success_criterion=success_criterion)
    print("UNAGGREGATED CLAIMED STATS")
    print(json.dumps(stats, indent=4))

    # print(' & '.join(f'${stats[c]['mean'] * 100:.1f}\\% \\pm {stats[c]['std'] * 100:.1f}\\%$' for c in categories))

    unclaimed_stats = plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax,
                                                               success_criterion=lambda x: RunAnalyser(
                                                                   x).success_criterion_satisfied(10))

    print("UNCLAIMED STATS")
    print(json.dumps(unclaimed_stats, indent=4))

    print(' & '.join(f'${unclaimed_stats[c]['mean'] * 100:.1f}\\%$' for c in categories))


if __name__ == "__main__":
    main()
