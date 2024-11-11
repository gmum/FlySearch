import os
import pathlib


def iterate_over_end_locations(root: pathlib.Path, example_paths):
    for path in example_paths:
        path = pathlib.Path(root / path)

        if path.is_dir():
            with open(path / "final_coords.txt") as f:
                end_coords = f.read().strip().replace("(", "").replace(")", "").split(", ")
                end_coords = tuple(map(float, end_coords))

                yield end_coords


def iterate_over_start_locations(root: pathlib.Path, example_paths):
    for path in example_paths:
        path = pathlib.Path(root / path)

        if path.is_dir():
            with open(path / "0_coords.txt") as f:
                end_coords = f.read().strip().replace("(", "").replace(")", "").split(", ")
                end_coords = tuple(map(float, end_coords))

                yield end_coords


def iterate_over_start_and_end_locations(root: pathlib.Path):
    example_paths = os.listdir(root)

    for start, end in zip(iterate_over_start_locations(root, example_paths),
                          iterate_over_end_locations(root, example_paths)):
        yield start, end

def l2_distance(start, end):
    return ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 + (end[2] - start[2]) ** 2) ** 0.5

def iterate_over_experiment_time_series(root: pathlib.Path, example_paths):
    for path in example_paths:
        path = pathlib.Path(root / path)

        if path.is_dir():
            filenames = os.listdir(path)
            coords_f = [filename for filename in filenames if filename.endswith("coords.txt") and filename != "start_rel_coords.txt"]

            def coord_key(coord):
                split = coord.split("_")

                if split[0] == "final":
                    return float('inf')
                else:
                    return int(split[0])

            coords_f = sorted(coords_f, key=coord_key)


            time_serie = []

            for coord_f in coords_f:
                with open(path / coord_f) as f:
                    coords = f.read().strip().replace("(", "").replace(")", "").split(", ")
                    coords = tuple(map(float, coords))

                    time_serie.append(coords)

            yield time_serie

def time_series_l2_distance_iterator(time_serie, correct_end_place):
    for subserie in time_serie:
        yield [l2_distance(state, correct_end_place) for state in subserie]

def get_l2_time_series(root: pathlib.Path, correct_end_place=(0, 0, 9)):
    example_paths = os.listdir(root)
    timeseries = list(iterate_over_experiment_time_series(root, example_paths))
    l2s = list(time_series_l2_distance_iterator(timeseries, correct_end_place))

    return l2s

def main():
    print(list(iterate_over_start_and_end_locations(pathlib.Path("../all_logs/g7tB"))))

    timeseries = list(iterate_over_experiment_time_series(pathlib.Path("../all_logs/g7tB"), os.listdir("../all_logs/g7tB")))

    print(list(time_series_l2_distance_iterator(timeseries, (0, 0, 9))))

if __name__ == "__main__":
    main()
