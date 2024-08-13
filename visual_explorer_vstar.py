import numpy as np
import cv2
import torchvision
import torch
import re
import string

from matplotlib import pyplot as plt
from openai import OpenAI

from abstract_conversation import Message, Conversation
from openai_conversation import OpenAITextMessage, OpenAIBase64ImageMessage, OpenAIConversation
from config import OPEN_AI_KEY


def add_grids_to_image(image: np.ndarray, splits: int, split_width: int) -> np.ndarray:
    print(image.shape)

    height, width = image.shape[:2]

    x_diff = width // splits
    y_diff = height // splits

    for i in range(split_width):
        image[:, i::x_diff, :] = 255
        image[i::y_diff, :, :] = 255

    return image


class OpenAIVisualVStarExplorer:
    number_glimpses = 5

    def get_starting_prompt(self):
        return f"""
I need you to provide an answer for a question about the image you see. However, the image provided will be in large resolution and the question in  act may very likely be about very minor details of the image. In fact, you may not be able to see object in question at all.  
        
You will be presented with options to answer the question. There is always a correct answer among these.
        
To make it easier for you, you can ask for specific parts (called glimpses) of the image in larger resolution by specifying (x, y) coordinates of the top-left and bottom-right corners of the rectangle you want to see.

For example, if you want to see the top-left corner of the image, you can specify (0, 0) and (0.25, 0.25) as the corners. Of course, you can also go wild and specify coordinates like (0.13, 0.72) and (0.45, 0.89) to see a different part of the image.

The first coordinate is horizontal, the second one is vertical. For example, to get the bottom-left corner of the image, you can specify (0.0, 0.75) and (0.25, 1). To help you out with coordinates, a white grid has been added to the image. Each line is roughly at 20% mark of the image's height or width (starting from 0, then through 0.2 and so on).

Using the same format, please specify the coordinates of the next rectangle you want to see or choose to answer the question. You also MUST specify your reasoning after each decision, as this is beneficial for LLMs, such as you. Put your reasoning in < and >.

YOUR COMMENTS MUST BE PUT IN < AND >. NOTHING ELSE SHOULD BE IN THESE BRACKETS. DO NOT PUT COORDINATES IN THESE BRACKETS.

You can request at most {self.number_glimpses - 1} glimpses.

OUTPUT FORMAT: (x1, y1) and (x2, y2) OR ANSWER: (your guess).
        
To answer, you will copy the entire text of the option you think is correct. Do not copy the letter meant to represent option's position.

Do not copy the "Researcher speaks" or "Model speaks" parts of the text. These are only cosmetic to convey the structure of the conversation.

Example: 

=== Researcher speaks ===
(Image of an airport terminal)
Question: Is the red suitcase on the left or right side of the man with Jagiellonian University t-shirt?
Options:
A. The red suitcase is on the left side of the man.
B. The red suitcase is on the right side of the man.

=== Model speaks ===
<I see many suitcases and people in the image, but they aren't zoomed in enough for me to discern details on them. I'll zoom on the bottom left corner of the image to see all people present.> (0.0, 0.75) and (0.25, 1)

=== Researcher speaks ===
(Glimpse of the bottom left corner of the image)

=== Model speaks ===
<I see only one red suitcase, but there are many men and I'm not sure about details on their T-shirts yet. I'll zoom on the suitcase and its horizontal surroundings to see who's wearing the Jagiellonian University T-shirt.> (0.05, 0.75) and (0.20, 0.85)

=== Researcher speaks ===
(Glimpse of the red suitcase and its surroundings)

=== Model speaks ===
<I have found a man with a logo of the Jagiellonian University on his T-shirt. The red suitcase is on his left side.> ANSWER: The red suitcase is on the left side of the man.
"""

    def get_classification_prompt(self, question: str, options: list[str]) -> str:
        return f"""
Question: {question}
Options:
{"\n".join(letter + ". " + option for letter, option in zip(string.ascii_uppercase, options))}

"""

    def __init__(self, conversation: Conversation, image: np.ndarray, question: str, options: list[str]) -> None:
        self.conversation = conversation
        self.image = add_grids_to_image(image, splits=5, split_width=5)
        self.glimpses = []
        self.glimpse_requests = []
        self.response = -1
        self.image_size = image.shape[:2]
        self.question = question
        self.options = options

    def step(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0, first=False) -> str:
        x1, y1, x2, y2 = self.convert_proportional_coords_to_pixel(x1, y1, x2, y2)

        glimpse = self.image[y1:y2, x1:x2, :]

        self.glimpses.append(glimpse)

        messages = [OpenAIBase64ImageMessage(cv2.imencode('.jpeg', glimpse)[1].tobytes(), 'jpeg') for glimpse in
                    self.glimpses]

        if first:
            messages = [OpenAITextMessage(self.get_starting_prompt()),
                        OpenAITextMessage(self.get_classification_prompt(self.question, self.options))] + messages

        self.conversation.send_messages(
            *messages
        )

        unfiltered = str(self.conversation.get_latest_message())
        unfiltered = unfiltered.replace("Model:", "").replace("model:",
                                                              "").strip()  # to avoid any funny business with the model's response
        return re.sub(r"<.*>", "", unfiltered, flags=re.S).replace("\n", "").strip()

    def convert_proportional_coords_to_pixel(self, x1, y1, x2, y2):
        height = self.image_size[0]
        width = self.image_size[1]

        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        return x1, y1, x2, y2

    def convert_str_coords_to_coords(self, coords: str) -> tuple:
        coords = coords.split("and")
        coords = [coord.strip() for coord in coords]

        x1, y1 = coords[0][1:-1].split(",")
        x2, y2 = coords[1][1:-1].split(",")

        x1 = float(x1.strip())
        y1 = float(y1.strip())
        x2 = float(x2.strip())
        y2 = float(y2.strip())

        return x1, y1, x2, y2

    def classify(self):
        response = self.step(first=True)

        # We are not subtracting one from self.number_glimpses because one step can be used for classification
        # If no classification is given, the model will respond "-1" and that won't be treated as a correct answer
        for _ in range(self.number_glimpses):
            if "answer" in response.lower():
                response = response.split(":")[1].strip()
                self.response = response
                return
            try:
                coords = self.convert_str_coords_to_coords(response)
            except:
                print("Invalid coordinates", response)
                return

            self.glimpse_requests.append(coords)

            response = self.step(*coords)

    def save_glimpse_boxes(self, filename: str) -> None:
        image_with_glimpses = self.image

        # Convert image to torch tensor
        image_with_glimpses = torch.tensor(image_with_glimpses).permute(2, 0, 1)

        for coords in self.glimpse_requests:
            x1, y1, x2, y2 = self.convert_proportional_coords_to_pixel(*coords)

            image_with_glimpses = torchvision.utils.draw_bounding_boxes(
                image_with_glimpses,
                torch.tensor([[x1, y1, x2, y2]]),
                width=5
            )

        # Convert image back to numpy array
        image_with_glimpses = image_with_glimpses.permute(1, 2, 0).numpy()

        # Save image
        cv2.imwrite(filename, image_with_glimpses)

    def save_glimpse_list(self, filename: str) -> None:
        _, axes = plt.subplots(1, len(self.glimpses))

        try:
            for glimpse, axe in zip(self.glimpses, axes):
                # Deal with matplotlib being funny
                glimpse = glimpse[:, :, ::-1]

                axe.imshow(glimpse)
                axe.axis("off")
        except TypeError:
            cv2.imwrite(filename, self.glimpses[0])
            plt.close()
            return

        plt.savefig(filename)
        plt.close()

    def save_glimpses_individually(self, filename_pref: str) -> None:
        try:
            for i, glimpse in enumerate(self.glimpses):
                cv2.imwrite(f"{filename_pref}_{i}.jpeg", glimpse)
        except:
            pass

    def save_unified_image(self, filename: str) -> None:
        image = np.zeros_like(self.image)

        glimpse_coords = [(0.0, 0.0, 1.0, 1.0)] + self.glimpse_requests

        for glimpse, coords in zip(self.glimpses, glimpse_coords):
            x_1, y_1, x_2, y_2 = self.convert_proportional_coords_to_pixel(*coords)
            diff_x = int(x_2 - x_1)
            diff_y = int(y_2 - y_1)

            glimpse = cv2.resize(glimpse, (diff_x, diff_y))

            image[y_1:y_2, x_1:x_2, :] = glimpse

        cv2.imwrite(filename, image)

    def get_response(self) -> int | str:
        return self.response


def main():
    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(client)

    # image = cv2.imread("imagenet-sample-images/n01616318_vulture.JPEG")
    # image = cv2.imread("imagenet-sample-images/n09835506_ballplayer.JPEG")
    # image = cv2.imread("imagenet-sample-images/n01632777_axolotl.JPEG")
    image = cv2.imread("sample_images/burger.jpeg")

    explorer = OpenAIVisualVStarExplorer(
        conversation,
        image,
        "What is written above 'McNuggets' on the box?",
        ["Kurczak", "Kaczka", "Kamyk", "Krowodrza Górka", "Krokodyl"]
    )

    explorer.classify()

    explorer.save_glimpse_boxes("glimpses.jpeg")
    explorer.save_glimpse_list("glimpse_list.jpeg")
    explorer.save_unified_image("unified_image.jpeg")
    print(explorer.get_response())


if __name__ == "__main__":
    main()
