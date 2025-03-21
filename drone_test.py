import os
import numpy as np
import pathlib

from matplotlib import pyplot as plt

from rl.city_fly_search_env import CityFlySearchEnv
from scenarios import ForestScenarioMapper, MimicScenarioMapper
from rl import ForestFlySearchEnv


def main():
    os.environ[
        "FOREST_BINARY_PATH"] = "/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"

    os.environ[
        "FONT_LOCATION"] = "/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

    os.environ[
        "CITY_BINARY_PATH"] = "/home/dominik/MyStuff/simulator/CitySample/Binaries/Linux/CitySample"

    fsm = MimicScenarioMapper(
        pathlib.Path("all_logs/forest-template")
    )

    env = ForestFlySearchEnv()

    images = []

    with env:
        for scenario, _ in zip(fsm.iterate_scenarios(), range(4)):
            obs, _ = env.reset(seed=None, options=scenario)
            opencv_image = obs["image"]
            pil_image = opencv_image[:, :, ::-1].copy()
            images.append(pil_image)

    figs, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 40))

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis("off")

    plt.show()

    csm = MimicScenarioMapper(
        pathlib.Path("all_logs/city-template")
    )
    env = CityFlySearchEnv()

    images = []

    with env:
        for scenario, _ in zip(csm.iterate_scenarios(), range(4)):
            obs, _ = env.reset(seed=None, options=scenario)
            opencv_image = obs["image"]
            pil_image = opencv_image[:, :, ::-1].copy()
            images.append(pil_image)

    figs, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 40))

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis("off")

    plt.show()

    csm = MimicScenarioMapper(
        pathlib.Path("all_logs/city-template")
    )
    env = CityFlySearchEnv()

    images = []

    with env:
        for scenario, _ in zip(csm.iterate_scenarios(), range(4)):
            obs, _ = env.reset(seed=None, options=scenario)
            opencv_image = obs["image"]
            pil_image = opencv_image[:, :, ::-1].copy()
            images.append(pil_image)

            obs2, *_ = env.step(
                {
                    "found": 0,
                    "coordinate_change": np.array([0, 20, 0]),
                })

            opencv_image = obs2["image"]
            pil_image = opencv_image[:, :, ::-1].copy()
            images.append(pil_image)

    figs, axs = plt.subplots(nrows=4, ncols=2, figsize=(40, 40))

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
