from scenarios import CityScenarioMapper


class DefaultCityAnomalyScenarioMapper(CityScenarioMapper):
    def __init__(self, drone_alt_min, drone_alt_max, alpha=0.5):
        super().__init__(
            object_probs={
                (
                    CityScenarioMapper.ObjectType.ANOMALY,
                ): 1.0
            },
            drone_z_rel_min=drone_alt_min * 100,
            drone_z_rel_max=drone_alt_max * 100,
            alpha=alpha,
        )
