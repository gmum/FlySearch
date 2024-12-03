from enum import Enum

class ForestScenarioMapper:
    class ObjectType(Enum):
        FIRE = 0
        TENT = 1
        TRASH = 2
        BUILDINGS = 3
        PEOPLE = 4
        ANOMALY = 5