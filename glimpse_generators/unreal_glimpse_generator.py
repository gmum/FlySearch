import subprocess
import os

from typing import Tuple

import cv2
from unrealcv import Client
from time import sleep
from PIL import Image

from misc.add_guardrails import dot_matrix_two_dimensional
from misc.cv2_and_numpy import opencv_to_pil, pil_to_opencv


class UnrealGlimpseGenerator:
    def __init__(self, host='localhost', port=9000, start_position=(3300.289, -26305.121, 0)):
        self.client = Client((host, port))
        self.start_position = start_position

        self._initialize_client()

    def _initialize_client(self):
        connection_result = self.client.connect()

        if not connection_result:
            raise ConnectionError("Failed to connect to UnrealCV server; is it running?")

        self.client.request('vget /unrealcv/status')
        self.client.request('vset /cameras/spawn')
        self.client.request('vset /camera/1/rotation -90 0 0')

        start_position = self.start_position

        self.client.request(
            f'vset /camera/0/location {start_position[0]} {start_position[1]} {start_position[2] + 10000}'
        )

    def disconnect(self):
        self.client.disconnect()

    def get_camera_image(self,
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0)) -> Image:
        start_position = self.start_position

        location = (start_position[0] + rel_position_m[0] * 100, start_position[1] + rel_position_m[1] * 100,
                    start_position[2] + rel_position_m[2] * 100)
        self.client.request(f'vset /camera/1/location {location[0]} {location[1]} {location[2]}')
        sleep(0.5)
        self.client.request('vget /camera/1/lit /tmp/camera.png')
        image = Image.open('/tmp/camera.png')
        image = image.resize((500, 500), Image.Resampling.BILINEAR)

        image = pil_to_opencv(image)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return opencv_to_pil(image)


class UnrealGridGlimpseGenerator(UnrealGlimpseGenerator):
    def __init__(self, splits_w: int, splits_h: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.splits_w = splits_w
        self.splits_h = splits_h

    def get_camera_image(self,
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0)) -> Image:
        img = super().get_camera_image(rel_position_m)
        img = pil_to_opencv(img)
        img = dot_matrix_two_dimensional(img, self.splits_w, self.splits_h)
        img = opencv_to_pil(img)

        return img


def main():
    generator = UnrealGridGlimpseGenerator(splits_w=5, splits_h=5)
    image = generator.get_camera_image((-50, -55, 100))
    image.show()

    generator.disconnect()


if __name__ == "__main__":
    main()
