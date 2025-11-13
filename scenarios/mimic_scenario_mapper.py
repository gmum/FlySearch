import json
import os
import pathlib

from rl.evaluation.configs.difficulty_levels import DifficultySettings
from scenarios.base_scenario_mapper import EpisodeCollectionMapper
from scenarios.city_scenario_mapper import CityScenarioMapper
from scenarios.forest_scenario_mapper import ForestScenarioMapper


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
def to_enum(
        value: str, scenario: str
) -> CityScenarioMapper.ObjectType | ForestScenarioMapper.ObjectType:
    if not is_enum(value):
        raise ValueError(f"Value {value} is not an enum")

    if scenario not in ["city", "forest"]:
        raise ValueError(f"Scenario {scenario} is not valid")

    value = value.strip().removeprefix("ObjectType.").lower()

    if scenario == "city":
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


def parse_scenario(scenario):
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


class MimicScenarioMapper(EpisodeCollectionMapper):
    def __init__(
            self, path: pathlib.Path, filter_str: str = "*", continue_from: int = 0
    ):
        self.path = path
        self.white_list = filter_str.strip().split(";")
        self.continue_from = continue_from
        self.scenarios = list(self.iterate_scenarios())

        object_type_cls = self.scenarios[0]["object_type"]

        # TODO: make better abstraction for this on refactor.
        self.difficulty = DifficultySettings.FS_2 if self.scenarios[0].get(
            "difficulty") == 'FS2' else DifficultySettings.FS_1
        self.is_city = type(object_type_cls) is CityScenarioMapper.ObjectType

        super().__init__({}, type(object_type_cls))

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, item: int):
        return self.scenarios[item]

    def __iter__(self):
        return iter(self.scenarios)

    def iterate_scenarios(self):
        entries = os.listdir(self.path)

        # Sort entries by episode number (and run number if available)
        try:
            entries = sorted(entries, key=lambda x: (int(x.split("_")[0]), int(x.split("r")[1])))
        except (IndexError, ValueError):
            entries = sorted(entries, key=lambda x: int(x))

        for entry in entries[self.continue_from:]:
            entry_path = self.path / entry
            if not os.path.isdir(entry_path):
                continue

            scenario = json.load(open(entry_path / "scenario_params.json"))
            object_name = scenario.get("passed_object_name", "")

            # Check if scenario matches whitelist
            if any(item in object_name or item == "*" for item in self.white_list):
                scenario = parse_scenario(scenario)
                scenario["drop"] = False
                scenario["i"] = int(entry.split("_")[0])
                yield scenario
