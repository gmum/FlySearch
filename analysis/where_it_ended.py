import pathlib

from matplotlib import pyplot as plt

from utils import l2_distance, iterate_over_start_and_end_locations

def plot_multiple_runs(start_end_loc_list, correct_end_place, ax):
    for start_end_locations in start_end_loc_list:
        start_heights = [start[2] for start, end in start_end_locations]

        ax.plot(start_heights,
                 [l2_distance(end, correct_end_place) for start, end in start_end_locations],
                 )

    return ax

def plot_baseline_for_run(start_end_locations, correct_end_place, ax, style="r--"):
    start_heights = [start[2] for start, end in start_end_locations]

    ax.plot(start_heights,
             [l2_distance(start, correct_end_place) for start, end in start_end_locations],
            style
    )

    return ax


def get_start_end_locations(root):
    return sorted(list(iterate_over_start_and_end_locations(pathlib.Path(root))), key=lambda x: x[0][2])

def plot_badgrid_vs_newgrid(ax):
    correct_end_place = (0, 0, 9)

    badgrid = get_start_end_locations("../all_logs/newgrid-gpt-4-badgrid")
    newgrid = get_start_end_locations("../all_logs/newgrid-gpt-3")

    ax = plot_multiple_runs(
        [badgrid, newgrid],
        correct_end_place, ax)

    plot_baseline_for_run(newgrid, correct_end_place, ax)

    ax.set_title("GPT-4o, directly above, badgrid vs newgrid")
    ax.set_xlabel("Starting height")
    ax.set_ylabel("Distance to target")

    ax.legend(["End of run distance, badgrid", "End of run distance, newgrid", "Distance to target at start"])

def plot_corner_vs_center(ax):
    correct_end_place = (0, 0, 9)

    corner = get_start_end_locations("../all_logs/newgrid-gpt-2")
    center = get_start_end_locations("../all_logs/newgrid-gpt-3")

    ax = plot_multiple_runs(
        [corner, center],
        correct_end_place, ax)

    plot_baseline_for_run(corner, correct_end_place, ax)
    plot_baseline_for_run(center, correct_end_place, ax, style="g--")

    ax.set_title("GPT-4o, directly above, corner vs center")
    ax.set_xlabel("Starting height")
    ax.set_ylabel("Distance to target")

    ax.legend(["End of run distance, corner", "End of run distance, center", "Distance to target at start, corner", "Distance to target at start, center"])

def plot_corner_forceful_vs_center_forceful(ax):
    correct_end_place = (0, 0, 9)

    corner = get_start_end_locations("../all_logs/newgrid-gpt-13-corner-incontext")
    center = get_start_end_locations("../all_logs/newgrid-gpt-11-closer-incontext")

    ax = plot_multiple_runs(
        [corner, center],
        correct_end_place, ax)

    plot_baseline_for_run(corner, correct_end_place, ax)
    plot_baseline_for_run(center, correct_end_place, ax, style="g--")

    ax.set_title("GPT-4o, directly above, corner vs center, forceful")
    ax.set_xlabel("Starting height")
    ax.set_ylabel("Distance to target")

    ax.legend(["End of run distance, corner", "End of run distance, center", "Distance to target at start, corner", "Distance to target at start, center"])

    return ax

def plot_corner_polite_vs_center_polite(ax):
    correct_end_place = (0, 0, 9)

    corner = get_start_end_locations("../all_logs/newgrid-gpt-14-corner-politealt")
    center = get_start_end_locations("../all_logs/newgrid-gpt-15-center-politealt")

    ax = plot_multiple_runs(
        [corner, center],
        correct_end_place, ax)

    plot_baseline_for_run(corner, correct_end_place, ax)
    plot_baseline_for_run(center, correct_end_place, ax, style="g--")

    ax.set_title("GPT-4o, directly above, corner vs center, polite + altitude")
    ax.set_xlabel("Starting height")
    ax.set_ylabel("Distance to target")

    # Set a horizontal line at 9
    ax.axhline(y=9, color='r', linestyle=':')

    ax.legend(["End of run distance, corner", "End of run distance, center", "Distance to target at start, corner", "Distance to target at start, center", "Distance to target from ground below it"])

    return ax

def plot_baseline_vs_closer_vs_forceful(ax):
    correct_end_place = (0, 0, 9)

    basic_run_start_end_locations = get_start_end_locations("../all_logs/newgrid-gpt-3")
    closer_run_start_end_locations = get_start_end_locations("../all_logs/newgrid-gpt-7-closer")
    incontext_run_start_end_locations = get_start_end_locations("../all_logs/newgrid-gpt-11-closer-incontext")

    ax = plot_multiple_runs(
        [basic_run_start_end_locations, closer_run_start_end_locations, incontext_run_start_end_locations],
        correct_end_place, ax)

    plot_baseline_for_run(basic_run_start_end_locations, correct_end_place, ax)

    ax.set_title("GPT-4o, directly above, different methods to get it to descend")
    ax.set_xlabel("Starting height")
    ax.set_ylabel("Distance to target")

    ax.legend(["End of run distance, baseline", "End of run distance, polite", "End of run distance, forceful + info", "Distance to target at start"])
def main():

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

    # plot_badgrid_vs_newgrid(ax[0])
    plot_corner_vs_center(ax[0])
    #plot_baseline_vs_closer_vs_forceful(ax[0])
    plot_corner_forceful_vs_center_forceful(ax[1])
    plot_corner_polite_vs_center_polite(ax[2])
    plt.show()

if __name__ == "__main__":
    main()
