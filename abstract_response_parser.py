from typing import Tuple
import re


class AbstractResponseParser:
    def __init__(self):
        pass

    def get_coordinates(self, response: str) -> Tuple[float, float, float, float] | None:
        pass

    def get_answer(self, response: str) -> str | None:
        pass

    def is_answer(self, response: str) -> bool:
        return self.get_coordinates(response) is None and self.get_answer(response) is not None

    def is_coordinate_request(self, response: str) -> bool:
        return self.get_coordinates(response) is not None

    def is_invalid(self, response: str) -> bool:
        return not self.is_answer(response) and not self.is_coordinate_request(response)


class SimpleResponseParser(AbstractResponseParser):
    def __init__(self):
        super().__init__()

    def filter_vlm_response(self, unfiltered: str) -> str:
        unfiltered = unfiltered.replace("Model:", "").replace("model:",
                                                              "").strip()  # to avoid any funny business with the model's response
        kinda_filtered = re.sub(r"<.*>", "", unfiltered, flags=re.S).replace("\n", "").strip()

        coordinates = re.match(r"\(.*\) and \(.*\)", kinda_filtered)

        if coordinates is not None:
            return coordinates.group(0)
        else:
            return kinda_filtered

    def get_coordinates(self, response: str) -> Tuple[float, float, float, float] | None:
        try:
            response = self.filter_vlm_response(response)
            response = response.lower()
            response = response.split("and")
            response = [coord.strip() for coord in response]

            x1, y1 = response[0][1:-1].split(",")
            x2, y2 = response[1][1:-1].split(",")

            x1 = float(x1.strip())
            y1 = float(y1.strip())
            x2 = float(x2.strip())
            y2 = float(y2.strip())

            return x1, y1, x2, y2
        except:
            return None

    def get_answer(self, response: str) -> str | None:
        try:
            response = self.filter_vlm_response(response)

            if "answer:" in response.lower():
                return response.split(":")[1].strip()
        except:
            return None
