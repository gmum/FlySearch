import pathlib
import os
import pandas as pd

from typing import List
from analysis.run import Run


def load_all_runs_from_a_dir(directory: pathlib.Path) -> List[Run]:
    return [Run(directory / run) for run in sorted(os.listdir(directory))]
