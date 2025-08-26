from scenarios import ForestScenarioMapper


class DefaultForestAnomalyScenarioMapper(ForestScenarioMapper):
    def __init__(self, drone_alt_min: int, drone_alt_max: int, alpha: float):
        super().__init__(
            object_probs={
                (
                    ForestScenarioMapper.ObjectType.ANOMALY,
                ): 1.0
            },
            x_min=15000,
            x_max=35000,
            y_min=15000,
            y_max=35000,
            z_min=0,
            z_max=1,
            drone_z_rel_min=drone_alt_min * 100,
            drone_z_rel_max=drone_alt_max * 100,
            alpha=alpha,
        )
