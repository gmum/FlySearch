import numpy as np
import cv2

from openai import OpenAI

from abstract_conversation import Message, Conversation
from openai_conversation import OpenAITextMessage, OpenAIBase64ImageMessage, OpenAIConversation
from config import OPEN_AI_KEY


class OpenAIVisualExplorer:
    glimpse_size = (32, 32)
    number_glimpses = 5
    prompt = f"""
        I need you to classify this image by using an imagenet class. However, it's been resized to {glimpse_size} pixels.
        You can ask for specific parts of the image in larger resolution by specifying (x, y) coordinates of the top-left 
        and bottom-right corners of the rectangle you want to see. Said rectangle will also be resized to {glimpse_size} pixels
        so the larger it is, the less detailed it will be. 
        
        For example, if you want to see the top-left corner of the image, you can specify (0, 0) and (0.25, 0.25) as the corners.
        Consequently, currently you are provided with the glimpse with coordinates (0, 0) and (1, 1).
        
        Of course, you can go wild and specify coordinates like (0.13, 0.72) and (0.45, 0.89) to see a different part of the image.
        Just do your best, it's a very important competition. Winner is determined by the most accurate classification AND the least 
        number of glimpses requested. If you win, you'll get a cake. I'll too. Otherwise I'll be fired. 
        
        Using the same format, please specify the coordinates of the next rectangle you want to see or choose to classify the image.
        DO NOT WRITE ANYTHING ELSE THAN THE COORDINATES OR YOUR CLASSIFICATION GUESS. I WILL BE FIRED IF YOU DO.
        
        WE WILL BOTH BE FIRED IF YOU REQUEST MORE THAN {number_glimpses} GLIMPSES.
        
        OUTPUT FORMAT: (x1, y1) and (x2, y2) OR CLASSIFICATION: <your guess>
        
        IF YOU DON'T USE IMAGENET CLASS, THE ENTIRE TEAM WILL BE FIRED.
    """

    def __init__(self, client: OpenAI, conversation: Conversation, image: np.ndarray) -> None:
        self.client = client
        self.conversation = conversation
        self.image = image
        self.glimpses = []

    def step(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0) -> str:
        height = self.image.shape[0]
        width = self.image.shape[1]

        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        glimpse = self.image[y1:y2, x1:x2]
        glimpse = cv2.resize(glimpse, self.glimpse_size)

        self.glimpses.append(glimpse)

        self.conversation.send_messages(
            OpenAITextMessage(self.prompt),
            *[OpenAIBase64ImageMessage(cv2.imencode('.jpeg', glimpse)[1].tobytes(), 'jpeg') for glimpse in
              self.glimpses]
        )

        return str(self.conversation.get_latest_message())

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
        response = self.step()

        for i in range(self.number_glimpses - 1):
            print("Response:", response)

            if "CLASSIFICATION" in response:
                return response

            coords = self.convert_str_coords_to_coords(response)
            response = self.step(*coords)


def main():
    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(client)

    image = cv2.imread("sample_images/burger.jpeg")

    explorer = OpenAIVisualExplorer(client, conversation, image)
    explorer.classify()


if __name__ == "__main__":
    main()
