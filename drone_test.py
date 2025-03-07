import os

from matplotlib import pyplot as plt
from scenarios import ForestScenarioMapper
from rl import ForestFlySearchEnv


def main():
    os.environ[
        "FOREST_BINARY_PATH"] = "/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"

    os.environ[
        "FONT_LOCATION"] = "/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

    fsm = ForestScenarioMapper(
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
        drone_z_rel_min=30 * 100,
        drone_z_rel_max=35 * 100,
        seed_min=1,
        seed_max=1000000000,
        scenarios_number=1,
    )

    scenario = fsm.create_random_scenario()

    env = ForestFlySearchEnv()

    with env:
        obs, _ = env.reset(scenario)
        opencv_image = obs["image"]
        pil_image = opencv_image[:, :, ::-1].copy()

    plt.imshow(pil_image)
    plt.show()


if __name__ == "__main__":
    main()
