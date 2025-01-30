import pathlib

from matplotlib import pyplot as plt

from analysis import Run, RunAnalyser, CriterionPlotter, load_all_runs_from_a_dir


def main():
    path = pathlib.Path("../all_logs/MC-0S-F-10G-GPT-CR")
    runs = load_all_runs_from_a_dir(path)
    plotter = CriterionPlotter(runs)

    runs_aggregated_per_type = plotter.get_runs_aggregated_per_type()

    print(runs_aggregated_per_type)

    fig, ax = plt.subplots(nrows=2, sharey=True, sharex=True)
    plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax[0])

    another_success_criterion = lambda x: RunAnalyser(x).success_criterion_satisfied(10)
    plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax[1],
                                             success_criterion=another_success_criterion)

    ax[0].set_title("MC + Claim")
    ax[1].set_title("MC, no claim")

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Object type")

    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("MC + Claim")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Starting height")

    per_height = plotter.get_runs_aggregated_per_height_bin()
    plotter.plot_accuracy_in_aggregated_runs(per_height, ax, threshold=10)

    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("MC + Claim / Fire")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Starting euclidean distance")

    per_mse_dist = plotter.aggregate_runs_per_function(
        lambda
            run: (run.start_position[0] ** 2 + run.start_position[1] ** 2 + run.start_position[2] ** 2) ** 0.5 // 10,
        lambda run: run.object_type == "FIRE"
    )

    plotter.plot_accuracy_in_aggregated_runs(per_mse_dist, ax, threshold=10)

    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("MC success")
    ax.set_ylabel("Accuracy")

    per_mse_dist = plotter.aggregate_runs_per_function(
        lambda run: 1,
    )

    plotter.plot_accuracy_in_aggregated_runs(per_mse_dist, ax, threshold=10)

    plt.show()

    def city_aggregation_function(run: Run):
        object_type = str(run.object_type).lower()

        if "car" in object_type:
            return "car"
        elif "pickup" in object_type:
            return "pickup"
        elif "truck" in object_type:
            return "truck"
        return object_type

    fig, ax = plt.subplots()

    per_cls_city_dict = plotter.aggregate_runs_per_function(city_aggregation_function)
    plotter.plot_accuracy_in_aggregated_runs(per_cls_city_dict, ax)

    plt.show()


if __name__ == "__main__":
    main()
