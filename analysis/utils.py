import pathlib
import os

from analysis.run import Run


def load_all_runs_from_a_dir(directory: pathlib.Path):
    return [Run(directory / run) for run in os.listdir(directory)]
