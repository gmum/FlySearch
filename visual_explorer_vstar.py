import typing

import numpy as np
import cv2
import torchvision
import torch
import re
import string

from matplotlib import pyplot as plt
from openai import OpenAI

from abstract_conversation import Conversation, Role
from config import OPEN_AI_KEY
from add_guardrails import dot_matrix_two_dimensional
from cv2_and_numpy import opencv_to_pil, pil_to_opencv
from prompt_generation import get_starting_prompt_for_vstar_explorer, get_classification_prompt_for_vstar_explorer
from image_glimpse_generator import ImageGlimpseGenerator


def add_grids_to_image(image: np.ndarray, splits: int, split_width: int) -> np.ndarray:
    return dot_matrix_two_dimensional(image, splits, splits)


class VisualVStarExplorer:
    def __init__(self,
                 conversation: Conversation,
                 image: np.ndarray,
                 question: str,
                 options: list[str],
                 glimpse_generator: ImageGlimpseGenerator,
                 number_glimpses: int = 5
                 ) -> None:
        self.conversation = conversation
        self.image = add_grids_to_image(image, splits=5, split_width=5)
        self.response = -1
        self.image_size = image.shape[:2]
        self.question = question
        self.options = options
        self.glimpse_generator = glimpse_generator
        self.glimpse_requests = [(0.0, 0.0, 1.0, 1.0)]
        self.number_glimpses = number_glimpses

    def filter_vlm_response(self, unfiltered: str) -> str:
        unfiltered = unfiltered.replace("Model:", "").replace("model:",
                                                              "").strip()  # to avoid any funny business with the model's response
        return re.sub(r"<.*>", "", unfiltered, flags=re.S).replace("\n", "").strip()

    def step(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0, first=False) -> str:
        glimpse = self.glimpse_generator.get_glimpse(x1, y1, x2, y2)

        self.conversation.begin_transaction(Role.USER)

        if first:
            self.conversation.add_text_message(get_starting_prompt_for_vstar_explorer(self.number_glimpses))
            self.conversation.add_text_message(
                get_classification_prompt_for_vstar_explorer(self.question, self.options))

        for subglimpse in glimpse:
            pil_subglimpse = opencv_to_pil(subglimpse)
            self.conversation.add_image_message(pil_subglimpse)

        self.conversation.commit_transaction(send_to_vlm=True)

        unfiltered = str(self.conversation.get_latest_message()[1])

        return self.filter_vlm_response(unfiltered)

    @staticmethod
    def convert_str_coords_to_coords(coords: str) -> tuple:
        coords = coords.split("and")
        coords = [coord.strip() for coord in coords]

        x1, y1 = coords[0][1:-1].split(",")
        x2, y2 = coords[1][1:-1].split(",")

        x1 = float(x1.strip())
        y1 = float(y1.strip())
        x2 = float(x2.strip())
        y2 = float(y2.strip())

        return x1, y1, x2, y2

    def answer(self):
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

    def get_glimpse_requests(self) -> list[tuple[float, float, float, float]]:
        return self.glimpse_requests

    def get_response(self) -> int | str:
        return self.response


def main():
    from openai_conversation import OpenAIConversation
    from image_glimpse_generator import GridImageGlimpseGenerator

    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(
        client,
        model_name="gpt-4o",
    )

    image = cv2.imread("sample_images/burger.jpeg")
    glimpse_generator = GridImageGlimpseGenerator(image, 5)

    explorer = VisualVStarExplorer(
        conversation,
        image,
        "What is written above 'McNuggets' on the box?",
        ["Kurczak", "Kaczka", "Kamyk", "Krowodrza Górka", "Krokodyl"],
        glimpse_generator
    )

    explorer.answer()

    # explorer.save_glimpse_boxes("glimpses.jpeg")
    # explorer.save_glimpse_list("glimpse_list.jpeg")
    # explorer.save_unified_image("unified_image.jpeg")
    print(explorer.get_response())

    from visualization import ExplorationVisualizer
    visualizer = ExplorationVisualizer(explorer.get_glimpse_requests(), explorer.glimpse_generator)

    visualizer.save_glimpse_list_figure("debug/glimpse_list.jpeg")
    visualizer.save_glimpses_individually("debug/glimpse")
    visualizer.save_glimpse_boxes("debug/glimpse_boxes.jpeg")


if __name__ == "__main__":
    main()
