from navigators import AbstractDroneNavigator, RecklessFlyingException
from response_parsers.xml_drone_response_parser import XMLDroneResponseParser


class GridDroneNavigator(AbstractDroneNavigator):
    def __init__(self):
        self.response_parser = XMLDroneResponseParser()

    def get_new_position(self, current_position: tuple[int, int, int], response: str, throw_if_reckless=False) -> tuple[
        int, int, int]:
        east_diff, north_diff, up_diff = self.get_diffs(response)

        current_west_east_axis, current_north_south_axis, current_up_down_axis = current_position

        # FOV = 90 degrees, so we can calculate what does it mean to fly recklessly

        if throw_if_reckless and abs(north_diff) > current_up_down_axis or abs(east_diff) > current_up_down_axis:
            raise RecklessFlyingException()

        # In this coordinate system, going east and north increases the value of the axis
        # In the internal one, going north decreases the value of the axis
        current_north_south_axis -= north_diff

        # Also, going east increases the value of the axis (yeah)
        current_west_east_axis += east_diff

        # Going up increases the value of the axis
        current_up_down_axis += up_diff

        return current_west_east_axis, current_north_south_axis, current_up_down_axis

    def get_diffs(self, response: str) -> tuple[int, int, int]:
        action_string = self.response_parser.get_basic_string(response)
        action_string = tuple(action_string.replace("(", "").replace(")", "").split(","))

        return int(action_string[0]), int(action_string[1]), int(action_string[2])

    def get_claim(self, response: str) -> bool:
        response = response.lower()
        response = response.replace(" ", "")

        return "found" in response
