from typing import Tuple
import cv2
import numpy as np

from glimpse_generators import UnrealClientWrapper
from time import sleep
from PIL import Image

from conversation import Role
from misc.add_guardrails import dot_matrix_two_dimensional_unreal
from misc.cv2_and_numpy import opencv_to_pil, pil_to_opencv


class OutOfBoundsException(Exception):
    pass


class UnrealGlimpseGenerator:
    def __init__(self, client: UnrealClientWrapper, start_position=(3300.289, -26305.121, 0)):
        self.client = client
        self.start_position = start_position

        self._initialize_client()

    # Sets the (0, 0, 0) point in our coordinate system
    def change_start_position(self, new_start_position: Tuple[float, float, float]):
        self.start_position = new_start_position

    def _initialize_client(self):

        self.client.request('vget /unrealcv/status')
        self.client.request('vset /cameras/spawn')
        self.client.request('vset /camera/1/rotation -90 0 0')
        self.client.request('vset /camera/1/size 1000 1000')

        self.reset_camera()

        while not self.is_unreal_ready():
            print("Unreal Glimpse Generator: Waiting for Unreal to give something meaningful...")
            sleep(0.5)

        print("Unreal Glimpse Generator: Unreal is ready to go!")

    def reset_camera(self):
        start_position = self.start_position

        self.client.request(
            f'vset /camera/1/location {start_position[0]} {start_position[1]} {start_position[2] + 10000}'
        )

    def disconnect(self):
        self.client.disconnect()

    def get_unreal_camera_coordinates(self) -> Tuple[float, float, float]:
        current = self.client.request('vget /camera/1/location')

        if type(current) == str:
            current = current.split(" ")
            assert len(current) == 3
            current = tuple(map(float, current))
        elif type(current) != tuple:
            raise ValueError("Unexpected type for current position received from UnrealCV. Got: ", type(current),
                             current)

        return current

    def get_relative_from_start(self):
        current = self.client.request('vget /camera/1/location')

        if type(current) == str:
            current = current.split(" ")
            assert len(current) == 3
            current = tuple(map(float, current))
        elif type(current) != tuple:
            raise ValueError("Unexpected type for current position received from UnrealCV. Got: ", type(current),
                             current)

        x, y, z = current

        x = x - self.start_position[0]
        x = x / 100

        y = y - self.start_position[1]
        y = y / 100

        z = z - self.start_position[2]
        z = z / 100

        return int(x), int(y), int(z)

    def wait_for_unreal_to_finish(self):
        while "false" in self.client.request("vget /camera/1/partition_loaded").lower():
            print("Unreal Glimpse Generator: Waiting for Unreal to finish loading the PARTITION, current status:",
                  self.client.request("vget /camera/1/partition_loaded"))
            sleep(0.5)

    def is_unreal_ready(self):
        img = self.__get_img(rel_position_m=(0, 0, 200), force_move=True)
        img = pil_to_opencv(img)

        blank = np.isclose(img, 0, atol=0.001).all()
        return not blank

    def __get_img(self, rel_position_m: Tuple[int, int, int] = (0, 0, 0), force_move=False) -> Image:
        start_position = self.start_position

        location = (start_position[0] + rel_position_m[0] * 100, start_position[1] + rel_position_m[1] * 100,
                    start_position[2] + rel_position_m[2] * 100)

        move_type = "location" if force_move else "moveto"

        self.client.request(f'vset /camera/1/{move_type} {location[0]} {location[1]} {location[2]}')
        if force_move:
            self.wait_for_unreal_to_finish()
        self.client.request('vget /camera/1/lit /tmp/camera.png')
        image = Image.open('/tmp/camera.png')
        image = image.resize((500, 500), Image.Resampling.BILINEAR)

        image = pil_to_opencv(image)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return opencv_to_pil(image)

    def get_camera_image(self,
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0), force_move=False) -> Image:
        return self.__get_img(rel_position_m, force_move=force_move)


class UnrealGridGlimpseGenerator(UnrealGlimpseGenerator):
    def __init__(self, splits_w: int, splits_h: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.splits_w = splits_w
        self.splits_h = splits_h

    def get_camera_image(self,
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0), force_move=False) -> Image:
        img = super().get_camera_image(rel_position_m, force_move=force_move)
        rel_position_m = self.get_relative_from_start()
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
                         rel_position_m: Tuple[int, int, int] = (0, 0, 0), force_move=False) -> Image:
        img = super().get_camera_image(rel_position_m, force_move=force_move)

        conversation = self.conversation_factory.get_conversation()
        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            f"Describe the image from the drone at an absolute altitude of {rel_position_m[2]}. Your description should be extremely detailed, and should include any objects, people, or other features that you see. If you see something of resemblance to {self.searched_obj}, mention it, specifying its approximate coordinates in the grid.")
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
