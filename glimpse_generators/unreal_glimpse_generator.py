import subprocess
import os

from typing import Tuple
from unrealcv import Client
from time import sleep
from PIL import Image


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
        return image.resize((224, 224), Image.Resampling.BILINEAR)


def main():
    generator = UnrealGlimpseGenerator()
    image = generator.get_camera_image((-50, -55, 100))
    image.show()

    generator.disconnect()


if __name__ == "__main__":
    main()
