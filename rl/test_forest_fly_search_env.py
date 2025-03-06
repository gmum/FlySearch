import pytest
import os

from matplotlib import pyplot as plt
from scenarios import ForestScenarioMapper
from rl import ForestFlySearchEnv


class TestForestFlySearchEnv:
    def test_acceptance(self):
        fsm = ForestScenarioMapper(
            object_probs={ForestScenarioMapper.ObjectType.FOREST_FIRE: 1.0},
            x_min=0,
            x_max=100,
            y_min=0,
            y_max=100,
            z_min=0,
            z_max=100,
            drone_z_rel_min=0,
            drone_z_rel_max=0,
            seed_min=0,
            seed_max=5,
            scenarios_number=1,
        )

        scenario = fsm.create_random_scenario()

        env = ForestFlySearchEnv()
        obs, _ = env.reset(scenario)

        image = obs["image"]

        plt.imshow(image)
        plt.show()
