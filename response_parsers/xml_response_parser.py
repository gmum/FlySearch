import re
from dataclasses import dataclass
from typing import Tuple


class ParsingError(ValueError):
    pass


@dataclass
class ModelResponse:
    found: bool = False
    move: Tuple[float, float, float] = (0, 0, 0)


XML_RESPONSE_PATTERN = re.compile(r'^.*?<action>(.*?)</action>.*$', flags=re.DOTALL)


def parse_xml_response(model_response: str) -> ModelResponse:
    model_response = model_response.lower().strip()
    match = XML_RESPONSE_PATTERN.match(model_response)

    if not match:
        if 'found' in model_response:
            # If the response contains 'found' but doesn't match the XML pattern, assume it's a found action
            return ModelResponse(found=True)

        raise ParsingError(f"Invalid XML response: {model_response}")

    action = match.group(1).strip()

    if 'found' in action:
        return ModelResponse(found=True)

    try:
        action = action.replace("(", "").replace(")", "")
        action = action.split(",")
        # east_diff, north_diff, up_diff
        action = float(action[0]), float(action[1]), float(action[2])
    except (ValueError, IndexError):
        raise ParsingError(f"Invalid action format: {action}")

    return ModelResponse(move=action)
