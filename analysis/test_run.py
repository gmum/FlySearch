import pathlib
import os

from analysis.run import Run


class TestRun:
    def test_coordinates_are_loaded_correctly(self):
        base_path = pathlib.Path("../all_logs/GPT4o-CityNew")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[0])

        coords = run.get_coords()

        assert len(coords) == 10
        assert coords[0] == (17, 10, 73)
        assert coords[1] == (-7, 10, 73)

    def test_params_are_loaded_correctly(self):
        base_path = pathlib.Path("../all_logs/GPT4o-CityNew")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[1])

        params = run.get_params()

        assert not run.forest_level
        assert run.seed == 382063673
        assert run.object_type == "WHITE_TRUCK"
        assert params["drone_rel_coords"] == "(-1.0, 6.0, 70.0)"
        assert run.model_claimed

    def test_comments_are_loaded_correctly(self):
        base_path = pathlib.Path("../all_logs/GPT4o-CityNew")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[2])

        comments = run.get_comments()

        print(comments)

        assert len(comments) == 7
        assert comments[0] == """<Reasoning>
I need to search systematically for the fire. I'll start by moving to the northeast corner to get a better view of the area.
</Reasoning>
<Action>(44, 44, 0)</Action>"""
        assert run.model_claimed

    def test_nothing_crashes_when_loading(self):
        base_path = pathlib.Path("../all_logs/GPT4o-CityNew")

        for run in os.listdir(base_path):
            run = Run(base_path / run)
            run.get_coords()
            run.get_params()
            run.get_comments()
            run.get_images()

    def test_model_claimed(self):
        base_path = pathlib.Path("../all_logs/GPT4o-CityNew")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[4])

        assert run.model_claimed

    def test_object_bbox_is_loaded(self):
        base_path = pathlib.Path("../all_logs/GPT4o-CityNew")
        runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

        run = Run(base_path / runs[0])

        object_bbox = run.get_object_bbox()

        assert object_bbox == (-91773.94, -50300.94, -191.62, -91242.81, -49769.81, 308.38)
