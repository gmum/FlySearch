import pytest

from navigators.grid_drone_navigator import GridDroneNavigator, RecklessFlyingException


class TestGridDroneNavigator:
    def test_get_new_position(self):
        navigator = GridDroneNavigator()
        current_position = (30, 20, 15)
        response = "<Action>(10, -40, -2)</Action>"
        assert navigator.get_new_position(current_position, response) == (40, 60, 13)

    def test_throw_if_reckless(self):
        navigator = GridDroneNavigator()
        current_position = (30, 20, 15)
        response = "<Action>(10, -40, -2)</Action>"

        with pytest.raises(RecklessFlyingException):
            navigator.get_new_position(current_position, response, throw_if_reckless=True)

        navigator.get_new_position(current_position, response, throw_if_reckless=False)

    def test_does_not_throw_if_not_reckless(self):
        navigator = GridDroneNavigator()
        current_position = (30, 20, 20)

        response = "<Action>(20, 20, 0)</Action>"

        assert navigator.get_new_position(current_position, response, throw_if_reckless=True) == (50, 0, 20)

        with pytest.raises(RecklessFlyingException):
            navigator.get_new_position(current_position, "<Action>(21, 20, 0)</Action>", throw_if_reckless=True)
