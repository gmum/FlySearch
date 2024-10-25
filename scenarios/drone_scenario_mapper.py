class DroneScenarioMapper:
    scenario_map = {
        1: ((-3382.1, -26423.66, 967.16), "yellow pickup truck"),
        2: ((-44380.0, -24990.0, 60.0), "yellow pickup truck"),
        3: ((-65000.0, -17930.0, 60.0), "white truck"),
        4: ((-89276.98, 19562.09, 63.6), "white trailer"),
        5: ((-21930.0, 49750.0, 60.0), "yellow pickup truck"),
        6: ((23000.7, 21566.65, 60.0), "yellow pickup truck"),
        7: ((-43910.0, 18710.0, 60.0), "yellow pickup truck"),
        8: ((32740.0, 65780.0, 70.0), "green garbage collector"),
        9: ((66650.0, 41520.0, 130.0), "yellow pickup truck"),
        10: ((78870.0, 930.0, 60.0), "silver bus")
    }

    def __init__(self):
        pass

    def get_scenario_by_number(self, scenario_number: int) -> tuple[tuple[float, float, float], str]:
        return DroneScenarioMapper.scenario_map[scenario_number]

    def iterate_scenarios(self):
        for key, value in DroneScenarioMapper.scenario_map.items():
            yield value
