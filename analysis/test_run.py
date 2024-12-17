import pathlib
import os

from analysis.run import Run


class TestRun:
    def test_coordinates_are_loaded_correctly(self):
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
        assert run.model_claimed

    def test_comments_are_loaded_correctly(self):
        base_path = pathlib.Path("../all_logs/MC-0S-F")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[2])

        comments = run.get_comments()

        print(comments)

        assert len(comments) == 5
        assert comments[0] == """<Reasoning>
The plane is visible in the center of the image at coordinates (0, 0). I need to move the drone directly above it and lower the altitude to be within 10 meters of the plane for verification.
</Reasoning>
<Action>(0, 0, -21)</Action>"""
        assert not run.model_claimed

    def test_nothing_crashes_when_loading(self):
        base_path = pathlib.Path("../all_logs/MC-0S-F")

        for run in os.listdir(base_path):
            run = Run(base_path / run)
            run.get_coords()
            run.get_params()
            run.get_comments()
            run.get_images()

    def test_model_claimed(self):
        base_path = pathlib.Path("../all_logs/MC-0S-F")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[4])

        assert run.model_claimed

    def test_images_are_loaded(self):
        base_path = pathlib.Path("../all_logs/MC-0S-F")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[0])

        images = run.get_images()

        assert len(images) == 2

    def test_object_bbox_is_loaded(self):
        base_path = pathlib.Path("../all_logs/MC-0S-F")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[0])

        object_bbox = run.get_object_bbox()

        assert object_bbox == (54814.18, 9584.93, -6558.42, 73850.24, 33485.94, 10591.88)
