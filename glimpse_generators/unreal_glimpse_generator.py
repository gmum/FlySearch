import subprocess
import os

from typing import Tuple

import cv2
from matplotlib.pyplot import connect
from unrealcv import Client
from time import sleep
from PIL import Image

from conversation import Role
from misc.add_guardrails import dot_matrix_two_dimensional_unreal
from misc.cv2_and_numpy import opencv_to_pil, pil_to_opencv


class UnrealGlimpseGenerator:
    def __init__(self, host='localhost', port=9000, start_position=(3300.289, -26305.121, 0)):
        self.host = host
        self.port = port
        self.client = None
        self.start_position = start_position

        self._initialize_client()

    def change_start_position(self, new_start_position: Tuple[int, int, int]):
        self.start_position = new_start_position

    def _initialize_client(self):
        connection_result = False

        for i in range(11):
            print(f"Trying to connect to UnrealCV server on port {self.port + i}")
            self.client = Client((self.host, self.port + i))
            connection_result = self.client.connect()

            if connection_result:
                break

        if not connection_result:
            raise ConnectionError("Failed to connect to UnrealCV server; is it running?")

        self.client.request('vget /unrealcv/status')
        self.client.request('vset /cameras/spawn')
        self.client.request('vset /camera/1/rotation -90 0 0')

        self.reset_camera()

    def reset_camera(self):
        start_position = self.start_position

        self.client.request(
            f'vset /camera/0/location {start_position[0]} {start_position[1]} {start_position[2] + 10000}'
        )

    def disconnect(self):
        self.client.disconnect()

    def get_relative_from_start(self):
        current = self.client.request('vget /camera/1/location')



    def get_camera_image(self,
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0)) -> Image:
        start_position = self.start_position

        location = (start_position[0] + rel_position_m[0] * 100, start_position[1] + rel_position_m[1] * 100,
                    start_position[2] + rel_position_m[2] * 100)
        self.client.request(f'vset /camera/1/moveto {location[0]} {location[1]} {location[2]}')
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
        img = dot_matrix_two_dimensional_unreal(img, self.splits_w, self.splits_h, drone_height=rel_position_m[2])
        img = opencv_to_pil(img)

        return img

class UnrealDescriptionGlimpseGenerator(UnrealGridGlimpseGenerator):
    def __init__(self, conversation_factory, searched_obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_factory = conversation_factory
        self.searched_obj = searched_obj

    def get_camera_image(self,
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0)) -> Image:
        img = super().get_camera_image(rel_position_m)

        conversation = self.conversation_factory.get_conversation()
        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(f"Describe the image from the drone at an absolute altitude of {rel_position_m[2]}. Your description should be extremely detailed, and should include any objects, people, or other features that you see. If you see something of resemblance to {self.searched_obj}, mention it, specifying its approximate coordinates in the grid.")
        conversation.add_image_message(img)
        conversation.commit_transaction(send_to_vlm=True)

        response = conversation.get_latest_message()[1]

        return img, response


def main():
    generator = UnrealGridGlimpseGenerator(splits_w=5, splits_h=5)
    image = generator.get_camera_image((-50, -55, 100))
    image.show()

    generator.disconnect()


if __name__ == "__main__":
    main()
