import numpy as np
import gymnasium as gym

from typing import Optional, Dict, Tuple, List
from glimpse_generators import UnrealClientWrapper, UnrealGlimpseGenerator, UnrealGridGlimpseGenerator
from misc import pil_to_opencv


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

    def _configure(self, options: Dict) -> None:
        raise NotImplementedError()

    def __init__(self, resolution: int = 500, max_altitude: int = 120):
        super().__init__()

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, shape=(resolution, resolution, 3), dtype=np.uint8),
                "altitude": gym.spaces.Box(0, max_altitude, shape=(1,), dtype=np.int32),
                "collision": gym.spaces.Discrete(2),  # 0 if no collision, 1 if collision
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "found": gym.spaces.Discrete(2),
                "coordinate_change": gym.spaces.Box(-max_altitude, max_altitude, shape=(3,), dtype=np.int32),
            }
        )

        self.client: UnrealClientWrapper = self.get_client()
        self.glimpse_generator: UnrealGlimpseGenerator = self.get_glimpse_generator(client=self.client)
        self.options: Optional[Dict] = None
        self.relative_position: Optional[np.ndarray] = None
        self.trajectory: Optional[List[np.ndarray]] = None
        self.started: bool = False

    '''
    Resets the environment according to the given seed and options. Should not be overridden by subclasses. It differs from the standard gym reset signature, as seed is passed as an element of the options dictionary. Furthermore, dictionary is a mandatory argument. Calls _configure with contents of the `options` dictionary. 
    
    Args:
        options: A dictionary of options to configure the environment. Some of these options should be environment-specific.
    
    Returns:
        First observation of the environment.
    '''

    def reset(self, options: Dict):
        if "seed" not in options:
            raise ValueError("Seed must be specified in options")

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

        return {
            "image": opencv_image,
            "altitude": altitude,
            "collision": 0,
        }, {}  # Observation and empty info

    def step(self, action: dict):
        if not self.started:
            raise UnitialisedEnvironmentException("Environment must be reset before calling step")

        if action["found"] == 1:
            # TODO/NOTE: In future, we may wanna have more semantics for finding the target
            self.started = False
            return {}, 0.0, True, False, {}  # Empty observation, no reward, terminated, no truncation, empty info

        new_ideal_position = self.relative_position + action["coordinate_change"]
        pil_image = self.glimpse_generator.get_camera_image(rel_position_m=new_ideal_position.tolist(),
                                                            force_move=False)

        new_real_position = np.array(self.glimpse_generator.get_relative_from_start())

        max_allowed_diff = 1
        max_diff = np.max(np.abs(new_ideal_position - new_real_position))
        crash = max_diff > max_allowed_diff

        opencv_image = pil_to_opencv(pil_image)

        self.trajectory.append(new_real_position)

        reward = self.get_reward()

        observation = {
            "image": opencv_image,
            "altitude": np.array([new_real_position[2]]),
            "collision": 1 if crash else 0,
        }

        return observation, reward, False, False, {}
