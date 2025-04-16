import os
import pathlib
import json
from typing import Dict, Any

from scenarios import CityScenarioMapper, ForestScenarioMapper, BaseScenarioMapper


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_float_tuple(value: str) -> bool:
    value = value.strip()
    if not value.startswith("(") or not value.endswith(")"):
        return False

    value = value[1:-1]

    for item in value.split(","):
        if not is_float(item):
            return False

    return True


def to_tuple(value: str) -> tuple:
    if not is_float_tuple(value):
        raise ValueError(f"Value {value} is not a tuple of floats")

    value = value[1:-1]

    return tuple(map(float, value.split(",")))


def is_bool(value: str) -> bool:
    value = value.strip().lower()
    return value == "true" or value == "false"


def to_bool(value: str) -> bool:
    if not is_bool(value):
        raise ValueError(f"Value {value} is not a boolean")

    return value.strip().lower() == "true"


def is_int(value: str) -> bool:
    return value.isdigit()


def is_enum(value: str) -> bool:
    return value.startswith("ObjectType.")


# I'm sorry
def to_enum(value: str, scenario: str) -> CityScenarioMapper.ObjectType | ForestScenarioMapper.ObjectType:
    if not is_enum(value):
        raise ValueError(f"Value {value} is not an enum")

    if scenario not in ["city", "forest"]:
        raise ValueError(f"Scenario {scenario} is not valid")

    value = value.strip().removeprefix("ObjectType.").lower()

    if scenario == "city":
        # Fix match because we use this stupid version of Python

        if value == "anomaly":
            return CityScenarioMapper.ObjectType.ANOMALY
        elif value == "police_car":
            return CityScenarioMapper.ObjectType.POLICE_CAR
        elif value == "beige_sport_car":
            return CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR
        elif value == "blue_sport_car":
            return CityScenarioMapper.ObjectType.BLUE_SPORT_CAR
        elif value == "red_sport_car":
            return CityScenarioMapper.ObjectType.RED_SPORT_CAR
        elif value == "white_sport_car":
            return CityScenarioMapper.ObjectType.WHITE_SPORT_CAR
        elif value == "road_construction_site":
            return CityScenarioMapper.ObjectType.ROAD_CONSTRUCTION_SITE
        elif value == "fire":
            return CityScenarioMapper.ObjectType.FIRE
        elif value == "black_pickup_truck":
            return CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK
        elif value == "green_pickup_truck":
            return CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK
        elif value == "red_pickup_truck":
            return CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK
        elif value == "white_pickup_truck":
            return CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK
        elif value == "crowd":
            return CityScenarioMapper.ObjectType.CROWD
        elif value == "large_trash_pile":
            return CityScenarioMapper.ObjectType.LARGE_TRASH_PILE
        elif value == "black_truck":
            return CityScenarioMapper.ObjectType.BLACK_TRUCK
        elif value == "white_truck":
            return CityScenarioMapper.ObjectType.WHITE_TRUCK
    elif scenario == "forest":
        if value == "forest_fire":
            return ForestScenarioMapper.ObjectType.FOREST_FIRE
        elif value == "campsite":
            return ForestScenarioMapper.ObjectType.CAMPSITE
        elif value == "trash_pile":
            return ForestScenarioMapper.ObjectType.TRASH_PILE
        elif value == "building":
            return ForestScenarioMapper.ObjectType.BUILDING
        elif value == "person":
            return ForestScenarioMapper.ObjectType.PERSON
        elif value == "anomaly":
            return ForestScenarioMapper.ObjectType.ANOMALY


class MimicScenarioMapper(BaseScenarioMapper):
    def __init__(self, path: pathlib.Path, filter_str: str = "*", continue_from: int = 0):
        self.path = path
        self.white_list = filter_str.strip().split(";")
        self.continue_from = continue_from
        self.scenarios = list(self.iterate_scenarios())

        object_type_cls = self.scenarios[0]["object_type"]

        super().__init__({}, type(object_type_cls))

    def parse_scenario(self, scenario):
        result = {}
        city = "regenerate_city" in scenario

        for key, value in scenario.items():
            if is_int(value):
                result[key] = int(value)
            elif is_float(value):
                result[key] = float(value)
            elif is_float_tuple(value):
                result[key] = to_tuple(value)
            elif is_bool(value):
                result[key] = to_bool(value)
            elif is_enum(value):
                result[key] = to_enum(value, "city" if city else "forest")
            else:
                result[key] = value

        return result

    def create_random_scenario(self, seed: int) -> Dict[str, Any]:
        return self.scenarios.pop(0)

    def empty(self):
        return len(self.scenarios) == 0

    def iterate_scenarios(self, duplicate_first=True):
        entries = os.listdir(self.path)
        entries = sorted(entries, key=lambda x: (int(x.split("_")[0]), int(x.split("r")[1])))[self.continue_from:]

        for entry in entries:
            if not os.path.isdir(self.path / entry):
                continue

            entry_number = int(entry.split("_")[0])
            scenario = json.load(open(self.path / entry / "scenario_params.json"))

            try:
                object_name = scenario["passed_object_name"]
            except KeyError:
                object_name = ""

            for white_list_item in self.white_list:
                if white_list_item in object_name or white_list_item == "*":
                    # This part is convoluted, so let me explain:
                    # 1. We need to duplicate first element, because we:
                    #   a. Need the engine to warm up
                    #   b. If we mimic, we want the first scenario.
                    # So, the solution is to duplicate the first scenario
                    # 2. We don't want parse a scenario dict twice, so the second if applies it only if we have already duplicated the first one
                    # Hence knowing that the current scenario was not parsed yet.

                    if duplicate_first:
                        scenario = self.parse_scenario(scenario)
                        scenario["drop"] = True
                        scenario["i"] = entry_number

                        yield scenario

                    if not duplicate_first:
                        scenario = self.parse_scenario(scenario)

                    scenario["drop"] = False
                    scenario["i"] = entry_number
                    duplicate_first = False

                    yield scenario
                    break


def main():
    mimic = MimicScenarioMapper(
        pathlib.Path("/home/anonymous/MyStuff/active-visual-gpt/all_logs/forest-template"), "*")

    for scenario in mimic.iterate_scenarios():
        print(scenario["i"])


if __name__ == "__main__":
    main()
