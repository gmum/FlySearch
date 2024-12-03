from enum import Enum

class DroneScenarioMapper:
    class ObjectType(Enum):
        YELLOW_PICKUP_TRUCK = 0
        WHITE_TRUCK = 1
        WHITE_TRAILER = 2
        GREEN_GARBAGE_COLLECTOR = 3
        SILVER_BUS = 4

    scenario_map = {
        1: ((-3382.1, -26423.66, 67.16), ObjectType.YELLOW_PICKUP_TRUCK),
        # The pickup truck is 900 units above that point (i. e. 9 meters)
        2: ((-44380.0, -24990.0, 60.0), ObjectType.YELLOW_PICKUP_TRUCK),
        3: ((-65000.0, -17930.0, 60.0), ObjectType.WHITE_TRUCK),
        4: ((-89276.98, 19562.09, 63.6), ObjectType.WHITE_TRAILER),
        5: ((-21930.0, 49750.0, 60.0), ObjectType.YELLOW_PICKUP_TRUCK),
        6: ((23000.7, 21566.65, 60.0), ObjectType.YELLOW_PICKUP_TRUCK),
        7: ((-43910.0, 18710.0, 60.0), ObjectType.YELLOW_PICKUP_TRUCK),
        8: ((32740.0, 65780.0, 70.0), ObjectType.GREEN_GARBAGE_COLLECTOR),
        9: ((66650.0, 41520.0, 130.0), ObjectType.YELLOW_PICKUP_TRUCK),
        10: ((78870.0, 930.0, 60.0), ObjectType.SILVER_BUS)
    }

    def __init__(self):
        pass

    def get_scenario_by_number(self, scenario_number: int) -> dict:
        return {
            "object_type": DroneScenarioMapper.scenario_map[scenario_number][1],
            "object_coords": DroneScenarioMapper.scenario_map[scenario_number][0],
            "drone_rel_coords": (0, 0, 50)
        }

    def iterate_scenarios(self):
        for i in range(1, len(DroneScenarioMapper.scenario_map) + 1):
            yield DroneScenarioMapper.scenario_map[i]



class YellowTruckScenarioMapper(DroneScenarioMapper):
    def __init__(self):
        super().__init__()

    def get_scenario_by_number(self, scenario_number: int) -> dict:
        if scenario_number == 1:
            return super().get_scenario_by_number(1)
        else:
            raise ValueError("This class only supports scenario 1")

    def iterate_scenarios(self):
        yield self.get_scenario_by_number(1)


class DroneScenarioMapperWithOffsets:
    def __init__(self,
                 min_x,
                 max_x,
                 step_x,
                 min_h,
                 max_h,
                 step_h,
                 min_y,
                 max_y,
                 step_y,
                 scenario_mapper):
        self.min_x = min_x
        self.max_x = max_x
        self.step_x = step_x

        self.min_h = min_h
        self.max_h = max_h
        self.step_h = step_h

        self.min_y = min_y
        self.max_y = max_y
        self.step_y = step_y

        if self.min_x > self.max_x:
            raise ValueError("min_x must be smaller or equal to max_x")

        if self.min_y > self.max_y:
            raise ValueError("min_y must be smaller or equal to max_y")

        if self.min_h > self.max_h:
            raise ValueError("min_h must be smaller or equal to max_h")

        if self.min_x is None or self.max_x is None or self.step_x is None:
            raise ValueError("min_x, max_x and step_x must be defined")

        if self.min_y is None or self.max_y is None or self.step_y is None:
            raise ValueError("min_y, max_y and step_y must be defined")

        if self.min_h is None or self.max_h is None or self.step_h is None:
            raise ValueError("min_h, max_h and step_h must be defined")

        if self.step_y is None or self.step_x is None or self.step_h is None:
            raise ValueError("step_y, step_x and step_h must be defined")

        self.scenario_mapper = scenario_mapper

    def iterate_scenarios(self):
        for scenario in self.scenario_mapper.iterate_scenarios():
            for x in range(self.min_x, self.max_x, self.step_x):
                for y in range(self.min_y, self.max_y, self.step_y):
                    for h in range(self.min_h, self.max_h, self.step_h):
                        scenario["drone_rel_coords"] = (x, y, h)
                        yield scenario
