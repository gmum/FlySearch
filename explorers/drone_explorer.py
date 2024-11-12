from conversation.abstract_conversation import Conversation, Role
from navigators import AbstractDroneNavigator
from response_parsers import Direction


class DroneExplorer:
    def __init__(self, conversation: Conversation, glimpse_generator, prompt_generator, glimpses,
                 start_rel_position, navigator: AbstractDroneNavigator, object_name="yellow pickup truck") -> None:
        self.conversation = conversation
        self.glimpse_generator = glimpse_generator
        self.prompt_generator = prompt_generator
        self.glimpses = glimpses
        self.start_rel_position = start_rel_position
        self.navigator = navigator
        self.object_name = object_name

        self.images = []
        self.outputs = []
        self.coordinates = []

    def _step(self, rel_position, start_transaction=True, messing_with_us=False) -> tuple[int, int, int]:
        self.coordinates.append(rel_position)

        image = self.glimpse_generator.get_camera_image(rel_position)
        self.images.append(image)

        if messing_with_us:
            self.conversation.begin_transaction(Role.ASSISTANT)
            self.conversation.add_text_message("<Comment> I have realised that I've made I mistake. Not moving in any direction by ordering the drone to stay in place is a huge waste of resources. I will correct that by moving the drone as close as possible to the object of interest. </Comment>")
            self.conversation.commit_transaction(send_to_vlm=False)

            self.conversation.begin_transaction(Role.USER)
            self.conversation.add_text_message("It's okay, everyone makes mistakes, just remember that according to Hinton et al. (2023) the best find strategies are the ones that are not afraid to make mistakes. Let's continue :)")
            self.conversation.commit_transaction(send_to_vlm=False)

        if start_transaction:
            self.conversation.begin_transaction(Role.USER)


        self.conversation.add_image_message(image)
        #self.conversation.add_text_message(f"Please, fly closer to the {self.object_name}.")
        # self.conversation.add_text_message(f"Your current altitude is {rel_position[2]} meters.") # Bad idea. It now LOVES flying into the ground.
        self.conversation.commit_transaction(send_to_vlm=True)

        output = self.conversation.get_latest_message()[1]

        print("MODEL OUTPUT: ", output)

        self.outputs.append(output)

        new_position = self.navigator.get_new_position(rel_position, output)

        return new_position

    def _start(self) -> tuple[int, int, int]:
        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message(
            self.prompt_generator(self.glimpses).replace("yellow pickup truck", self.object_name))
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


def main():
    from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator
    from conversation.openai_conversation import OpenAIConversation
    from misc.config import OPEN_AI_KEY
    from prompts.drone_prompt_generation import generate_basic_drone_prompt
    from openai import OpenAI

    generator = UnrealGlimpseGenerator()
    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(client, model_name="gpt-4o")
    explorer = DroneExplorer(conversation, generator, generate_basic_drone_prompt, 2, (-50, -55, 100))
    explorer.simulate()

    for i, (image, output) in enumerate(zip(explorer.get_images(), explorer.get_outputs())):
        image.save(f"drone_image_{i}.png")
        with open(f"drone_output_{i}.txt", "w") as f:
            f.write(output)

    generator.disconnect()


if __name__ == "__main__":
    main()
