from conversation.abstract_conversation import Conversation, Role
from response_parsers import Direction


class DroneExplorer:
    def __init__(self, conversation: Conversation, glimpse_generator, prompt_generator, glimpses,
                 start_rel_position, response_parser) -> None:
        self.conversation = conversation
        self.glimpse_generator = glimpse_generator
        self.prompt_generator = prompt_generator
        self.glimpses = glimpses
        self.start_rel_position = start_rel_position
        self.response_parser = response_parser

        self.images = []
        self.outputs = []
        self.coordinates = [start_rel_position]

    def _step(self, rel_position, start_transaction=True) -> tuple[int, int, int]:
        self.coordinates.append(rel_position)

        image = self.glimpse_generator.get_camera_image(rel_position)
        self.images.append(image)

        if start_transaction:
            self.conversation.begin_transaction(Role.USER)

        self.conversation.add_text_message(f'Current position: {rel_position[0]}, {rel_position[1]}, {rel_position[2]}')
        self.conversation.add_image_message(image)
        self.conversation.commit_transaction(send_to_vlm=True)

        output = self.conversation.get_latest_message()[1]

        print("MODEL OUTPUT: ", output)

        self.outputs.append(output)

        direction = self.response_parser.get_direction_from_response(output)
        distance = self.response_parser.get_distance_from_response(output)

        if direction == Direction.NORTH:
            return rel_position[0], rel_position[1] + distance, rel_position[2]
        elif direction == Direction.SOUTH:
            return rel_position[0], rel_position[1] - distance, rel_position[2]
        elif direction == Direction.EAST:
            return rel_position[0] + distance, rel_position[1], rel_position[2]
        elif direction == Direction.WEST:
            return rel_position[0] - distance, rel_position[1], rel_position[2]
        elif direction == Direction.UP:
            return rel_position[0], rel_position[1], rel_position[2] + distance
        elif direction == Direction.DOWN:
            return rel_position[0], rel_position[1], rel_position[2] - distance
        else:
            raise ValueError(f'Unknown direction: {direction}')

    def _start(self) -> tuple[int, int, int]:
        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message(self.prompt_generator(self.glimpses))
        return self._step(self.start_rel_position, start_transaction=False)

    def simulate(self):
        try:
            position = self._start()

            for _ in range(self.glimpses - 1):
                position = self._step(position)
        except Exception as e:
            print("Drone explorer failed", e)

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
