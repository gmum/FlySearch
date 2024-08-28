import re

from abstract_response_parser import AbstractResponseParser
from typing import Tuple


class XMLResponseParser(AbstractResponseParser):
    def __init__(self):
        super().__init__()

    def get_coordinates(self, response: str) -> Tuple[float, float, float, float] | None:
        response = response.lower()

        try:
            response = response.replace("\n", "")
            response = re.findall(r"<request>.*</request>", response, flags=re.S)[0]
            response = response.removeprefix("<request>")
            response = response.removesuffix("</request>")

            response = response.split("and")
            response = [coord.strip() for coord in response]

            x1, y1 = response[0][1:-1].split(",")
            x2, y2 = response[1][1:-1].split(",")

            x1 = float(x1.strip())
            y1 = float(y1.strip())
            x2 = float(x2.strip())
            y2 = float(y2.strip())

            return x1, y1, x2, y2
        except Exception as e:
            # print("Failed to parse coordinates", e, response)
            return None

    def get_answer(self, response: str) -> str | None:
        response = response.lower()

        try:
            response = response.replace("\n", "")
            response = re.findall(r"<answer>.*</answer>", response, flags=re.S)[0]
            response = response.removeprefix("<answer>")
            response = response.removesuffix("</answer>")

            return response
        except Exception as e:
            # print("Failed to get answer", e, response)
            return None
