from navigators import AbstractDroneNavigator
from response_parsers.xml_drone_response_parser import XMLDroneResponseParser


class RecklessFlyingException(Exception):
    pass


class GridDroneNavigator(AbstractDroneNavigator):
    def __init__(self):
        self.response_parser = XMLDroneResponseParser()

    def get_new_position(self, current_position: tuple[int, int, int], response: str, throw_if_reckless=False) -> tuple[
        int, int, int]:
        action_string = self.response_parser.get_basic_string(response)
        action_string = tuple(action_string.replace("(", "").replace(")", "").split(","))

        east_diff, north_diff, up_diff = int(action_string[0]), int(action_string[1]), int(action_string[2])

        current_west_east_axis, current_north_south_axis, current_up_down_axis = current_position

        # FOV = 90 degrees, so we can calculate what does it mean to fly recklessly

        if throw_if_reckless and abs(north_diff) > current_up_down_axis or abs(east_diff) > current_west_east_axis:
            raise RecklessFlyingException()

        # In this coordinate system, going east and north increases the value of the axis
        # In the internal one, going north decreases the value of the axis
        current_north_south_axis -= north_diff

        # Also, going east increases the value of the axis (yeah)
        current_west_east_axis += east_diff

        # Going up increases the value of the axis
        current_up_down_axis += up_diff

        return current_west_east_axis, current_north_south_axis, current_up_down_axis
