from scenarios import ForestScenarioMapper


class DefaultForestScenarioMapper(ForestScenarioMapper):
    def __init__(self, drone_alt_min: int, drone_alt_max: int, alpha: float = 0.5):
        super().__init__(
            x_min=15000,
            x_max=35000,
            y_min=15000,
            y_max=35000,
            z_min=0,
            z_max=1,
            drone_z_rel_min=drone_alt_min * 100,
            drone_z_rel_max=drone_alt_max * 100,
            object_probs={
                (ForestScenarioMapper.ObjectType.PERSON,
                 ForestScenarioMapper.ObjectType.FOREST_FIRE,
                 ForestScenarioMapper.ObjectType.TRASH_PILE,
                 ForestScenarioMapper.ObjectType.CAMPSITE,
                 ForestScenarioMapper.ObjectType.BUILDING): 1.0
            },
            alpha=alpha,
        )
