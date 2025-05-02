import numpy as np
import gymnasium as gym

from typing import Optional, Dict, Tuple, List

from glimpse_generators import UnrealClientWrapper, UnrealGlimpseGenerator, UnrealGridGlimpseGenerator
from misc import pil_to_opencv

from scenarios import get_classes_to_object_classes, classes_to_images
from scenarios.object_classes import BaseObjectClass


class DroneCannotSeeTargetException(Exception):
    pass


class UnitialisedEnvironmentException(Exception):
    pass


class BaseFlySearchEnv(gym.Env):
    def get_client(self) -> UnrealClientWrapper:
        raise NotImplementedError()

    @staticmethod
    def get_glimpse_generator(client: UnrealClientWrapper) -> UnrealGlimpseGenerator:
        return UnrealGridGlimpseGenerator(client=client, splits_w=6, splits_h=6)

    def get_reward(self) -> float:
        return 0.0

    def _configure(self, options: Dict) -> None:
        """
        Configures the environment with the given options. Should be overridden by sublasses. Should set agent's and target's locations in the engine.

        If the drone cannot see the target, it should raise DroneCannotSeeTargetException.

        Args:
            options: A dictionary of options to configure the environment.

        Returns:
            None

        Throws:
            DroneCannotSeeTargetException: If the options specify a scenario where the drone cannot see the target. This condition should be checked by the subclass using the game engine.

        """
        raise NotImplementedError()

    def __init__(self, resolution: int = 500, max_altitude: int = 120, throw_if_hard_config: bool = True,
                 give_class_image: bool = False):
        super().__init__()

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2

        obs_dict = {
            "image": gym.spaces.Box(0, 255, shape=(resolution, resolution, 3), dtype=np.uint8),
            "altitude": gym.spaces.Box(0, max_altitude, shape=(1,), dtype=np.int32),
            "collision": gym.spaces.Discrete(2),  # 0 if no collision, 1 if collision
        }

        if give_class_image:
            obs_dict["class_image"] = gym.spaces.Box(0, 255, shape=(resolution, resolution, 3), dtype=np.uint8)

        self.observation_space = gym.spaces.Dict(
            obs_dict
        )

        self.action_space = gym.spaces.Dict(
            {
                "found": gym.spaces.Discrete(2),
                "coordinate_change": gym.spaces.Box(-max_altitude, max_altitude, shape=(3,), dtype=np.int32),
            }
        )

        self.resolution = resolution

        self.client: Optional[UnrealClientWrapper] = None
        self.glimpse_generator: Optional[UnrealGlimpseGenerator] = None
        self.classes_to_ids: Optional[Dict] = None

        self.options: Optional[Dict] = None
        self.relative_position: Optional[np.ndarray] = None
        self.trajectory: Optional[List[np.ndarray]] = None

        self.started: bool = False
        self.resources_initialized: bool = False
        self.throw_if_hard_config: bool = throw_if_hard_config

        self.give_class_image: bool = give_class_image

    def set_throw_if_hard_config(self, throw_if_hard_config: bool) -> None:
        self.throw_if_hard_config = throw_if_hard_config

    def __enter__(self):
        self.client = self.get_client()
        self.glimpse_generator = self.get_glimpse_generator(client=self.client)
        self.classes_to_ids = get_classes_to_object_classes(self.client)

        self.resources_initialized = True

        return self

    def __exit__(self, *_, **__):
        self.client.disconnect()
        self.resources_initialized = False

        return False

    def get_object_bbox(self):
        return self.client.request(f"vget /object/{self.options['object_id']}/bounds")

    def reset(self, seed: Optional[int] = None, options: Dict = None):
        """
        Resets the environment according to the given seed and options. Should not be overridden by subclasses. Furthermore, dictionary is a mandatory argument. Calls _configure with contents of the `options` dictionary.

        Args:
            seed: A seed for the environment. If specified, overrides the seed in the options.
            options: A dictionary of options to configure the environment. Some of these options should be environment-specific. Must be specified. Default is None only for compatibility with the gymnasium API.

        Returns:
            First observation of the environment.
        """

        if options is None:
            raise ValueError("Options must be specified")

        if seed is not None:
            options["seed"] = seed

        if "seed" not in options:
            raise ValueError("Seed must be specified by options or as a seed argument!")

        if not self.resources_initialized:
            raise UnitialisedEnvironmentException(
                "Environment must be entered before calling reset. Use a `with` statement. Otherwise, the simulator itself won't be running.")

        super().reset(seed=options["seed"], options=None)

        self.options = options
        self._configure(options)

        self.relative_position = np.array(self.glimpse_generator.get_relative_from_start())
        self.trajectory = [self.relative_position]

        pil_image = self.glimpse_generator.get_camera_image(rel_position_m=self.relative_position.tolist(),
                                                            force_move=True)
        opencv_image = pil_to_opencv(pil_image)

        altitude = np.array([self.relative_position[2]])

        self.started = True

        obs = {
            "image": opencv_image,
            "altitude": altitude,
            "collision": 0,
        }

        if self.give_class_image:
            class_image = classes_to_images[self.options["object_type"]]
            class_image = class_image.resize((self.resolution, self.resolution))
            opencv_class_image = pil_to_opencv(class_image)
            obs["class_image"] = opencv_class_image

        return obs, {"real_position": self.relative_position,
                     "object_bbox": self.get_object_bbox()}

    def step(self, action: dict):
        if not self.started:
            raise UnitialisedEnvironmentException("Environment must be reset before calling step")

        if action["found"] == 1:
            # TODO/NOTE: In future, we may wanna have more semantics for finding the target
            self.started = False
            return {}, 0.0, True, False, {
                "real_position": self.relative_position,
                "object_bbox": self.get_object_bbox()}  # Empty observation, no reward, terminated, no truncation, info with relative position and bbox

        coordinate_change = action["coordinate_change"]

        if isinstance(coordinate_change, tuple):
            if len(coordinate_change) != 3:
                raise ValueError("Coordinate change must be a tuple of 3 integers.")
            coordinate_change = np.array(coordinate_change)

        coordinate_change[1] = -coordinate_change[1]  # The rest of the code inverts the y axis. Yeah, I know.

        new_ideal_position = self.relative_position + coordinate_change
        pil_image = self.glimpse_generator.get_camera_image(rel_position_m=new_ideal_position.tolist(),
                                                            force_move=False)

        new_real_position = np.array(self.glimpse_generator.get_relative_from_start())

        max_allowed_diff = 1
        max_diff = np.max(np.abs(new_ideal_position - new_real_position))
        crash = max_diff > max_allowed_diff

        opencv_image = pil_to_opencv(pil_image)

        self.relative_position = new_real_position
        self.trajectory.append(new_real_position)

        reward = self.get_reward()

        observation = {
            "image": opencv_image,
            "altitude": np.array([new_real_position[2]]),
            "collision": 1 if crash else 0,
        }

        if self.give_class_image:
            class_image = classes_to_images[self.options["object_type"]]
            class_image = class_image.resize((self.resolution, self.resolution))
            opencv_class_image = pil_to_opencv(class_image)
            observation["class_image"] = opencv_class_image

        return observation, reward, False, False, {"real_position": new_real_position,
                                                   "object_bbox": self.get_object_bbox()}

    # Bunch of utility functions

    def hide_all_movable_objects(self) -> None:
        for object_class in self.classes_to_ids.values():
            if isinstance(object_class, BaseObjectClass):
                object_class.hide_all_objects()

    # Sets the camera in a given location and asks for camera image, ensuring that the map is loaded
    def load_map(self, x, y, z, drone_rel_x_semi, drone_rel_y_semi, drone_rel_z_semi) -> None:
        # Asking glimpse generator for a glimpse will effectively load the map in a given location
        self.glimpse_generator.change_start_position((x, y, z))
        self.glimpse_generator.reset_camera()
        self.glimpse_generator.get_camera_image((drone_rel_x_semi, drone_rel_y_semi, drone_rel_z_semi), force_move=True)

    def rel_to_real(self, x, y, z, x_rel, y_rel, z_rel):
        x_rel *= 100
        y_rel *= 100
        z_rel *= 100

        return x + x_rel, y + y_rel, z + z_rel
