import numpy as np
import cv2
import torchvision
import torch
import re

from matplotlib import pyplot as plt
from openai import OpenAI

from abstract_conversation import Message, Conversation
from openai_conversation import OpenAITextMessage, OpenAIBase64ImageMessage, OpenAIConversation
from config import OPEN_AI_KEY


class OpenAIVisualExplorer:
    glimpse_size = (32, 32)
    image_size = (224, 224)
    number_glimpses = 5

    def get_prompt(self):
        return f"""
        I need you to classify this image by using an imagenet class {"NUMBER" if self.set_label else "STRING"}. However, it's been resized to {self.glimpse_size} pixels.
        You can ask for specific parts of the image in larger resolution by specifying (x, y) coordinates of the top-left 
        and bottom-right corners of the rectangle you want to see. Said rectangle will also be resized to {self.glimpse_size} pixels
        so the larger it is, the less detailed it will be. 
        
        For example, if you want to see the top-left corner of the image, you can specify (0, 0) and (0.25, 0.25) as the corners.        
        Of course, you can also go wild and specify coordinates like (0.13, 0.72) and (0.45, 0.89) to see a different part of the image.
                        
        Using the same format, please specify the coordinates of the next rectangle you want to see or choose to classify the image.
        You also MUST specify your reasoning after each decision, as this is beneficial for LLMs, such as you. Put your reasoning in < and >.
        YOUR COMMENTS MUST BE PUT IN < AND >. NOTHING ELSE SHOULD BE IN THESE BRACKETS.
        
        Classify when you're reasonably certain about the specific Imagenet class. Note that there are many classes that mamy seem similar, so you need to be very precise.
                        
        You can request at most {self.number_glimpses} glimpses.
        
        OUTPUT FORMAT: (x1, y1) and (x2, y2) OR CLASSIFICATION: (your guess).
        
    """

    def __init__(self, conversation: Conversation, image: np.ndarray, set_label=False) -> None:
        self.conversation = conversation
        self.image = cv2.resize(image, self.image_size)
        self.glimpses = []
        self.glimpse_requests = []
        self.response = -1
        self.set_label = set_label
        self.prompt = self.get_prompt()

    def step(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0, first=False) -> str:
        x1, y1, x2, y2 = self.convert_proportional_coords_to_pixel(x1, y1, x2, y2)

        glimpse = self.image[y1:y2, x1:x2, :]
        glimpse = cv2.resize(glimpse, self.glimpse_size)

        self.glimpses.append(glimpse)

        messages = [OpenAIBase64ImageMessage(cv2.imencode('.jpeg', glimpse)[1].tobytes(), 'jpeg') for glimpse in
                    self.glimpses]

        if first:
            messages = [OpenAITextMessage(self.prompt)] + messages

        self.conversation.send_messages(
            *messages
        )

        unfiltered = str(self.conversation.get_latest_message())
        return re.sub(r"<.*>", "", unfiltered)

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
            if "classification" in response.lower():
                response = response.split(":")[1].strip()
                if self.set_label:
                    try:
                        self.response = int(response)
                    except:
                        print("Invalid response")
                else:
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
    image = cv2.imread("imagenet-sample-images/n01632777_axolotl.JPEG")
    # image = cv2.imread("sample_images/burger.jpeg")

    explorer = OpenAIVisualExplorer(conversation, image)
    explorer.classify()

    explorer.save_glimpse_boxes("glimpses.jpeg")
    explorer.save_glimpse_list("glimpse_list.jpeg")
    explorer.save_unified_image("unified_image.jpeg")
    print(explorer.get_response())


if __name__ == "__main__":
    main()
