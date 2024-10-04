from enum import Enum


class Direction(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    UP = 5
    DOWN = 6


class AbstractDroneResponseParser:
    def get_direction_from_response(self, response: str) -> Direction:
        pass

    def get_distance_from_response(self, response: str) -> int:
        pass
    