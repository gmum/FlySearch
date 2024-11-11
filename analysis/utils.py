import os
import pathlib


def iterate_over_end_locations(root: pathlib.Path, example_paths):
    for path in example_paths:
        path = pathlib.Path(root / path)

        print(path)

        if path.is_dir():
            with open(path / "final_coords.txt") as f:
                end_coords = f.read().strip().replace("(", "").replace(")", "").split(", ")
                end_coords = tuple(map(float, end_coords))

                yield end_coords


def iterate_over_start_locations(root: pathlib.Path, example_paths):
    for path in example_paths:
        path = pathlib.Path(root / path)

        print(path)

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


def main():
    print(list(iterate_over_start_and_end_locations(pathlib.Path("../all_logs/g7tB"))))


if __name__ == "__main__":
    main()
