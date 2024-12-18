from typing import Tuple

from conversation.abstract_conversation import Conversation, Role
from navigators import AbstractDroneNavigator
from response_parsers import Direction
from PIL import Image


class DroneExplorer:
    def __init__(self, conversation: Conversation, glimpse_generator, prompt_generator, glimpses,
                 start_rel_position, navigator: AbstractDroneNavigator, object_name="yellow pickup truck",
                 incontext=False, abandon_image_on_cue=True) -> None:
        self.conversation = conversation
        self.glimpse_generator = glimpse_generator
        self.prompt_generator = prompt_generator
        self.glimpses = glimpses
        self.start_rel_position = start_rel_position
        self.navigator = navigator
        self.object_name = object_name
        self.incontext = incontext
        self.abandon_image_on_cue = abandon_image_on_cue

        self.images = []
        self.outputs = []
        self.coordinates = []

    def _incontext_step(self):
        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message("Your current altitude is 21 meters.")
        self.conversation.add_image_message(Image.open("/net/people/plgrid/plgdmatuszek/active-visual-gpt/data/2.png"))
        self.conversation.commit_transaction(send_to_vlm=False)

        self.conversation.begin_transaction(Role.ASSISTANT)
        self.conversation.add_text_message(
            "<Comment> I clearly see a yellow pickup truck that I am centered on in both of my axes. I can however adjust my altitude to get even closer to it. Since my altitude is 21 meters, I will adjust it by 10 meters in a calm and professional manner. </Comment> <Action> (0, 0, -10) </Action>")
        self.conversation.commit_transaction(send_to_vlm=False)

        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message("Your current altitude is 11 meters.")
        self.conversation.add_image_message(Image.open("/net/people/plgrid/plgdmatuszek/active-visual-gpt/data/3.png"))
        self.conversation.commit_transaction(send_to_vlm=False)

        self.conversation.begin_transaction(Role.ASSISTANT)
        self.conversation.add_text_message(
            "<Comment> I am now at 11 meters and all I can see is the yellow pickup truck due to my proximity. I have achieved the goal. </Comment> <Action> FOUND </Action>")
        self.conversation.commit_transaction(send_to_vlm=False)

        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message("Very vell, let's move on to another example.")
        self.conversation.commit_transaction(send_to_vlm=False)

    def _step(self, rel_position, start_transaction=True, messing_with_us=False) -> tuple[float, float, float]:

        print("Rel position assuming no crashes: ", rel_position)

        ideal_rel_position = rel_position

        glimpse = self.glimpse_generator.get_camera_image(rel_position)
        rel_position = self.glimpse_generator.get_relative_from_start()

        print("Real rel position: ", rel_position)

        max_allowed_diff = 1

        diffX = abs(ideal_rel_position[0] - rel_position[0])
        diffY = abs(ideal_rel_position[1] - rel_position[1])
        diffZ = abs(ideal_rel_position[2] - rel_position[2])

        crash_detected = False

        if diffX > max_allowed_diff or diffY > max_allowed_diff or diffZ > max_allowed_diff:
            crash_detected = True

        self.coordinates.append(rel_position)

        abandon_sending_image = False
        cue = None

        if isinstance(glimpse, tuple):
            image, cue = glimpse
            abandon_sending_image = self.abandon_image_on_cue
        else:
            image = glimpse

        self.images.append(image)

        if messing_with_us:
            self.conversation.begin_transaction(Role.USER)
            self.conversation.add_text_message("Please, fly closer to the object of interest.")
            self.conversation.commit_transaction(send_to_vlm=False)

        if start_transaction:
            self.conversation.begin_transaction(Role.USER)

        if crash_detected:
            self.conversation.add_text_message(
                "Emergency stop; you've flown too close to something and would have hit it.")

        if cue is not None:
            self.conversation.add_text_message(cue)

        if not abandon_sending_image:
            self.conversation.add_image_message(image)

        self.conversation.add_text_message(f"Your current altitude is {rel_position[2]} meters.")
        self.conversation.commit_transaction(send_to_vlm=True)

        output = self.conversation.get_latest_message()[1]

        print("MODEL OUTPUT: ", output)

        self.outputs.append(output)

        new_position = self.navigator.get_new_position(rel_position, output)

        return new_position

    def _start(self) -> tuple[float, float, float]:
        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message(
            self.prompt_generator(self.glimpses).replace("yellow pickup truck", self.object_name))

        if self.incontext:
            self.conversation.commit_transaction(send_to_vlm=False)
            self._incontext_step()
            self.conversation.begin_transaction(Role.USER)

        return self._step(self.start_rel_position, start_transaction=False)

    def simulate(self) -> tuple[int, int, int]:
        position = self.start_rel_position
        try:
            old_position = position
            position = self._start()
            messing_with_us = (position == old_position)

            for _ in range(self.glimpses - 1):
                old_position = position
                position = self._step(position, messing_with_us=messing_with_us)
                messing_with_us = (position == old_position)

        except Exception as e:
            print("Drone explorer failed", e)

        return position

    def get_images(self):
        return self.images

    def get_outputs(self):
        return self.outputs

    def get_coords(self):
        return self.coordinates
