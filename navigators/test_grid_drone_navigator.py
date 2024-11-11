from navigators.grid_drone_navigator import GridDroneNavigator


class TestGridDroneNavigator:
    def test_get_new_position(self):
        navigator = GridDroneNavigator()
        current_position = (30, 20, 15)
        response = "<Action>(10, -40, -2)</Action>"
        assert navigator.get_new_position(current_position, response) == (-10, 10, 13)
