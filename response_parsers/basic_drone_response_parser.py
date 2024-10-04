from response_parsers.abstract_drone_response_parser import AbstractDroneResponseParser, Direction


class BasicDroneResponseParser(AbstractDroneResponseParser):
    def get_direction_from_response(self, response: str) -> Direction:
        response = response.lower()

        direction = response.split(' ')[1]

        if direction == 'north':
            return Direction.NORTH
        elif direction == 'south':
            return Direction.SOUTH
        elif direction == 'east':
            return Direction.EAST
        elif direction == 'west':
            return Direction.WEST
        elif direction == 'up':
            return Direction.UP
        elif direction == 'down':
            return Direction.DOWN
        else:
            raise ValueError(f'Unknown direction: {direction}')

    def get_distance_from_response(self, response: str) -> int:
        return int(response.split(' ')[2].split(';')[0])
