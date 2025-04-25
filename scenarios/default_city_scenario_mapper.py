from scenarios import CityScenarioMapper


class DefaultCityScenarioMapper(CityScenarioMapper):
    def __init__(self, drone_alt_min, drone_alt_max, alpha=0.5):
        super().__init__(
            object_probs={
                (CityScenarioMapper.ObjectType.POLICE_CAR,
                 CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR,
                 CityScenarioMapper.ObjectType.BLUE_SPORT_CAR,
                 CityScenarioMapper.ObjectType.RED_SPORT_CAR,
                 CityScenarioMapper.ObjectType.WHITE_SPORT_CAR,
                 CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.WHITE_TRUCK,
                 CityScenarioMapper.ObjectType.BLACK_TRUCK
                 ): 0.2,
                (
                    CityScenarioMapper.ObjectType.ROAD_CONSTRUCTION_SITE,
                ): 0.2,
                (
                    CityScenarioMapper.ObjectType.FIRE,
                ): 0.2,
                (
                    CityScenarioMapper.ObjectType.CROWD,
                ): 0.2,
                (
                    CityScenarioMapper.ObjectType.LARGE_TRASH_PILE,
                ): 0.2
            },
            drone_z_rel_min=drone_alt_min * 100,
            drone_z_rel_max=drone_alt_max * 100,
            alpha=alpha,
        )
