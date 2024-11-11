import pathlib

from matplotlib import pyplot as plt

from utils import get_l2_time_series

def main():
    plt.subplot(1, 2, 1)

    for time_serie in get_l2_time_series(pathlib.Path("../all_logs/newgrid-gpt-2")):
        print("Time serie length:", len(time_serie))
        plt.plot(range(len(time_serie)), time_serie)

    plt.subplot(1, 2, 2)

    for time_serie in get_l2_time_series(pathlib.Path("../all_logs/newgrid-gpt-3")):
        print("Time serie length:", len(time_serie))
        plt.plot(range(len(time_serie)), time_serie)

    plt.show()

if __name__ == "__main__":
    main()