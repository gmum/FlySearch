import pathlib

from matplotlib import pyplot as plt

from utils import l2_distance, iterate_over_start_and_end_locations


def main():
    root = pathlib.Path("../all_logs/newgrid-gpt-3")
    correct_end_place = (0, 0, 9)

    start_end_locations = sorted(list(iterate_over_start_and_end_locations(root)), key=lambda x: x[0][2])
    start_heights = [start[2] for start, end in start_end_locations]

    print(start_end_locations)
    print(start_heights)

    plt.plot(start_heights,
             [l2_distance(start, correct_end_place) for start, end in start_end_locations],
             )

    plt.plot(start_heights,
             [l2_distance(end, correct_end_place) for start, end in start_end_locations],
             )

    # root = pathlib.Path("../all_logs/g7tB")
    root = pathlib.Path("../all_logs/newgrid-gpt-7-closer")
    intern_start_end_locations = sorted(list(iterate_over_start_and_end_locations(root)),
                                        key=lambda x: x[0][2])
    intern_start_end_locations = [(start, end) for start, end in intern_start_end_locations if start[2]]
    start_heights = [start[2] for start, end in intern_start_end_locations]

    plt.plot(start_heights,
             [l2_distance(end, correct_end_place) for start, end in intern_start_end_locations],
             )

    #plt.axvline(x=29, color='r', label='Car can be seen in the corner', ls='--')

    plt.xlabel("Start height")
    plt.ylabel("Euclidean distance")

    plt.legend(
        ["Euclidean distance on start", "Euclidean distance on end / GPT4o above", "Euclidean distance on end / GPT4o above, telling it to get closer",
         #"Car can be seen in the corner from here"
         ])

    plt.title("Start height and end euclidean distance")

    plt.show()


if __name__ == "__main__":
    main()
