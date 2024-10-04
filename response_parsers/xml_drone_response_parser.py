import re

from response_parsers.abstract_drone_response_parser import AbstractDroneResponseParser, Direction
from response_parsers.basic_drone_response_parser import BasicDroneResponseParser


class XMLDroneResponseParser(AbstractDroneResponseParser):
    def get_basic_string(self, response: str) -> str:
        response = response.lower()

        response = re.findall(r"<action>.*</action>", response, flags=re.S)[0]
        response = response.removeprefix("<action>")
        response = response.removesuffix("</action>")

        return response

    def get_direction_from_response(self, response: str) -> Direction:
        basic_string = self.get_basic_string(response)
        return BasicDroneResponseParser().get_direction_from_response(basic_string)

    def get_distance_from_response(self, response: str) -> int:
        basic_string = self.get_basic_string(response)
        return BasicDroneResponseParser().get_distance_from_response(basic_string)
