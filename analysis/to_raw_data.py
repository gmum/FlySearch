import pathlib

from analysis import convert_runs_from_dir_to_pandas


def main():
    run_name = "MC-0S-F-2"
    directory = pathlib.Path(f"../all_logs/{run_name}")

    df = convert_runs_from_dir_to_pandas(directory)

    df.to_csv(f"raw_csv/{run_name}.csv", index=False)


if __name__ == "__main__":
    main()
