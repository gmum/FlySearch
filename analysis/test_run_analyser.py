from analysis.run_analyser import RunAnalyser


class TestRunAnalyser:
    class MockRun:
        def __init__(self, end_position, object_bbox):
            self.coords = [end_position]
            self.object_bbox = object_bbox
            self.object_type = "car"

        def get_coords(self):
            return self.coords

        def get_object_bbox(self):
            return self.object_bbox

        @property
        def end_position(self):
            return self.coords[-1]

    def test_object_visible(self):
        run = self.MockRun((0, 0, 100), (-1000, -1000, 0, 1000, 1000, 300))

        run_analyser = RunAnalyser(run)

        assert run_analyser.object_visible()

    def test_object_within_altitude_threshold(self):
        run = self.MockRun((0, 0, 100), (-1000, -1000, 0, 1000, 1000, 9000))

        run_analyser = RunAnalyser(run)

        assert run_analyser.drone_within_altitude_threshold(10)
        assert run_analyser.success_criterion_satisfied(10, check_claimed=False)

    def test_object_offset(self):
        run = self.MockRun((-100, 100, 101), (-30000, -5000, -500, -300, 1000, 9500))

        run_analyser = RunAnalyser(run)

        assert run_analyser.object_visible()
        assert run_analyser.drone_within_altitude_threshold(10)
        assert run_analyser.success_criterion_satisfied(10, check_claimed=False)

    def test_euclidean_distance_to_zero(self):
        run = self.MockRun((0, 3, 4), (-1000, -1000, 0, 1000, 1000, 300))

        run_analyser = RunAnalyser(run)

        assert run_analyser.get_euclidean_distance() == 5
