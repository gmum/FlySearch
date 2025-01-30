import pathlib
import pandas as pd

from analysis import load_all_runs_from_a_dir, Run, RunAnalyser


def convert_runs_from_dir_to_pandas(directory: pathlib.Path) -> pd.DataFrame:
    runs = load_all_runs_from_a_dir(directory)

    def run_to_dict(run: Run):
        to_return = run.params

        analyser = RunAnalyser(run)

        to_return["object_visible"] = analyser.object_visible()
        to_return["success_criterion"] = analyser.success_criterion_satisfied()
        to_return["drone_within_threshold"] = analyser.drone_within_altitude_threshold(10)
        to_return["model_claimed"] = run.model_claimed

        object_coords = run.real_object_coords
        del to_return["object_coords"]

        to_return["object_x"] = object_coords[0]
        to_return["object_y"] = object_coords[1]
        to_return["object_z"] = object_coords[2]

        drone_rel_coords = run.start_position
        del to_return["drone_rel_coords"]

        to_return["drone_rel_x"] = drone_rel_coords[0]
        to_return["drone_rel_y"] = drone_rel_coords[1]
        to_return["drone_rel_z"] = drone_rel_coords[2]

        for i, position in enumerate(run.coords):
            to_return[f"coord_x_{i}"] = position[0]
            to_return[f"coord_y_{i}"] = position[1]
            to_return[f"coord_z_{i}"] = position[2]

        to_return["object_type"] = " ".join(run.object_type.lower().split("_"))

        return to_return

    return pd.DataFrame(
        [run_to_dict(run) for run in runs]
    )
