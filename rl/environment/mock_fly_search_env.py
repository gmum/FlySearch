import numpy as np

from glimpse_generators import UnrealClientWrapper, UnrealGlimpseGenerator, UnrealGridGlimpseGenerator
from rl.environment import BaseFlySearchEnv


class MockClient(UnrealClientWrapper):
    def __init__(self):
        pass

    def disconnect(self):
        pass

    def request(self, *args, **kwargs):
        pass


class MockGlimpseGenerator(UnrealGridGlimpseGenerator):
    def __init__(self):
        pass

    def get_relative_from_start(self):
        return 42, 42, 42

    def get_camera_image(self, rel_position_m=(0, 0, 0), force_move=False):
        return np.zeros((512, 512, 3))


class MockFlySearchEnv(BaseFlySearchEnv):
    def __init__(self):
        super().__init__()

    def get_client(self):
        return MockClient()

    def _configure(self, options) -> None:
        pass

    @staticmethod
    def get_glimpse_generator(client):
        return MockGlimpseGenerator()
