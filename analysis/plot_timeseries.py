import pathlib

from matplotlib import pyplot as plt

from utils import get_l2_time_series

def plot_time_series(path, ax, title):
    s = get_l2_time_series(pathlib.Path(path))

    for time_serie in s:
        ax.plot(range(len(time_serie)), time_serie)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to target")

def plot_closer_prompting_strategies(ax):
    plot_time_series("../all_logs/newgrid-gpt-3", ax[0], "GPT-4o, directly above")
    plot_time_series("../all_logs/newgrid-gpt-7-closer", ax[1], "GPT-4o, directly above, prompting to get closer")
    plot_time_series("../all_logs/newgrid-gpt-11-closer-incontext", ax[2], "GPT-4o, directly above, forcefully winning the argument")

def plot_corner_vs_center(ax):
    plot_time_series("../all_logs/newgrid-gpt-2", ax[0], "GPT-4o, corner")
    plot_time_series("../all_logs/newgrid-gpt-3", ax[1], "GPT-4o, center")


def plot_badgrid_vs_newgrid(ax):
    plot_time_series("../all_logs/newgrid-gpt-4-badgrid", ax[0], "GPT-4o, directly above, badgrid")
    plot_time_series("../all_logs/newgrid-gpt-3", ax[1], "GPT-4o, directly above, newgrid")
    plot_time_series("../all_logs/newgrid-gpt-11-closer-incontext", ax[2], "GPT-4o, directly above, newgrid, forcefully winning the argument")

def plot_politeness(ax):
    plot_time_series("../all_logs/newgrid-gpt-14-corner-politealt", ax[0], "GPT-4o, corner, polite + altitude")
    plot_time_series("../all_logs/newgrid-gpt-15-center-politealt", ax[1], "GPT-4o, center, polite + altitude")


def main():
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    plot_politeness(axs)
    plt.show()

    exit()

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)

    plot_closer_prompting_strategies(axs)
    plt.show()

    figs, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    plot_corner_vs_center(axs)
    plt.show()

    figs, axs = plt.subplots(1, 3, sharex=True, sharey=True)

    plot_badgrid_vs_newgrid(axs)
    plt.show()

if __name__ == "__main__":
    main()