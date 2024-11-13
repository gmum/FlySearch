import pathlib

from matplotlib import pyplot as plt

from utils import get_l2_time_series

def main():
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)

    s1 = get_l2_time_series(pathlib.Path("../all_logs/newgrid-gpt-3"))
    s2 = get_l2_time_series(pathlib.Path("../all_logs/newgrid-gpt-7-closer"))
    s3 = get_l2_time_series(pathlib.Path("../all_logs/newgrid-gpt-11-closer-incontext"))

    # Set titles
    axs[0].set_title("GPT-4o, directly above")
    axs[1].set_title("GPT-4o, directly above, prompting to get closer")
    axs[2].set_title("GPT-4o, directly above, forcefully winning the argument")

    for time_serie in s1:
        axs[0].plot(range(len(time_serie)), time_serie)

    for time_serie in s2:
        axs[1].plot(range(len(time_serie)), time_serie)

    for time_serie in s3:
        axs[2].plot(range(len(time_serie)), time_serie)

    plt.show()

if __name__ == "__main__":
    main()