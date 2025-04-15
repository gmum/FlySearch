import numpy as np
import cv2

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
    def __init__(self, start_unreal_coords=(50000.0, 50000.0, 1000.0), current_relative_coords=(0.0, 0.0, 0.0)):
        self.start_unreal_coords = start_unreal_coords
        self.current_relative_coords = current_relative_coords

    def get_relative_from_start(self):
        return [
            int(self.current_relative_coords[0]),
            int(self.current_relative_coords[1]),
            int(self.current_relative_coords[2]),
        ]

    def get_camera_image(self, rel_position_m=(0, 0, 0), force_move=False):
        image = cv2.imread("/home/dominik/MyStuff/active-visual-gpt/data/golden_retriever.png")
        image = cv2.resize(image, (512, 512))
        image = cv2.resize(image, (500, 500))

        self.current_relative_coords = rel_position_m

        return image


class MockFlySearchEnv(BaseFlySearchEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_client(self):
        return MockClient()

    def _configure(self, options) -> None:
        pass

    @staticmethod
    def get_glimpse_generator(client):
        return MockGlimpseGenerator()

    def get_object_bbox(self):
        return '0.0 0.0 0.0 0.0 0.0 0.0'
