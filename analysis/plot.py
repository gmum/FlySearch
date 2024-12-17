import pathlib

from matplotlib import pyplot as plt

from analysis import Run, RunAnalyser, CriterionPlotter, load_all_runs_from_a_dir


def main():
    path = pathlib.Path("../all_logs/MC-0S-F")
    runs = load_all_runs_from_a_dir(path)
    plotter = CriterionPlotter(runs)

    runs_aggregated_per_type = plotter.get_runs_aggregated_per_type()

    print(runs_aggregated_per_type)

    fig, ax = plt.subplots(nrows=2, sharey=True, sharex=True)
    plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax[0])

    another_success_criterion = lambda x: RunAnalyser(x).maciek_criterion_satisfied(10)
    plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_type, ax[1],
                                             success_criterion=another_success_criterion)

    ax[0].set_title("MC + Claim")
    ax[1].set_title("MC, no claim")

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Object type")

    plt.show()


if __name__ == "__main__":
    main()
