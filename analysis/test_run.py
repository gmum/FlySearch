import pathlib
import os

from analysis.run import Run

class TestRun:
    def test_coordinates_are_loaded_correctly(self):
        print(os.getcwd())
        base_path = pathlib.Path("../all_logs/MC-0S-F")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[0])

        coords = run.get_coords()

        assert len(coords) == 2
        assert coords[0] == (3.0, 29.0, 70.0)
        assert coords[1] == (2.0, 4.0, 9.0)

    def test_params_are_loaded_correctly(self):
        base_path = pathlib.Path("../all_logs/MC-0S-F")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[1])

        params = run.get_params()

        assert run.forest_level
        assert run.seed == 357
        assert run.object_type == "FIRE"
        assert params["drone_rel_coords"] == "(-20, 0, 56)"