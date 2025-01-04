from navigators import RecklessFlyingException
from navigators.abstract_drone_navigator import AbstractDroneNavigator
from response_parsers import AbstractDroneResponseParser, Direction


class TrivialDroneNavigator(AbstractDroneNavigator):
    def __init__(self, response_parser: AbstractDroneResponseParser):
        self.response_parser = response_parser

    def get_new_position(self, current_position: tuple[int, int, int], response: str,
                         throw_if_reckless: bool = False) -> tuple[
        int, int, int]:

        direction = self.response_parser.get_direction_from_response(response)
        distance = self.response_parser.get_distance_from_response(response)

        if direction != Direction.UP and direction != Direction.DOWN and throw_if_reckless:
            if abs(distance) > current_position[2]:
                raise RecklessFlyingException()

        if direction == Direction.NORTH:
            return current_position[0], current_position[1] - distance, current_position[2]
        elif direction == Direction.SOUTH:
            return current_position[0], current_position[1] + distance, current_position[2]
        elif direction == Direction.EAST:
            return current_position[0] + distance, current_position[1], current_position[2]
        elif direction == Direction.WEST:
            return current_position[0] - distance, current_position[1], current_position[2]
        elif direction == Direction.UP:
            return current_position[0], current_position[1], current_position[2] + distance
        elif direction == Direction.DOWN:
            return current_position[0], current_position[1], current_position[2] - distance
        else:
            raise ValueError(f'Unknown direction: {direction}')
