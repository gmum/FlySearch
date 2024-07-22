import torch
import torchvision
import cv2
import numpy as np
import os
import pathlib
import json

from tqdm import tqdm
from openai import OpenAI
from time import sleep

from visual_explorer import OpenAIVisualExplorer
from abstract_conversation import Message, Conversation
from openai_conversation import OpenAIConversation
from config import OPEN_AI_KEY
from correctness_checker import check_validity_of_answer


def from_pil_to_opencv(image):
    return np.array(image)[:, :, ::-1].copy()


def main():
    imagenet = torchvision.datasets.ImageNet(
        root="/home/dominik/imagenet",
        split="val",
        transform=from_pil_to_opencv
    )

    approx_test_cases = 1
    dataset_size = len(imagenet)
    step = dataset_size // approx_test_cases
    start = 1

    subset = torch.utils.data.Subset(imagenet, range(start, dataset_size, step))
    conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY))

    correct_answers = 0
    total_answers = 0

    bar = tqdm(enumerate(subset), total=len(subset))

    for i, (image, label) in bar:
        pathlib.Path(f"test_logs/{i}").mkdir(exist_ok=True)

        explorer = OpenAIVisualExplorer(conversation, image)
        explorer.classify()

        explorer.save_glimpse_boxes(f"test_logs/{i}/glimpses.png")
        explorer.save_glimpse_list(f"test_logs/{i}/glimpses_list_{i}.png")
        explorer.save_unified_image(f"test_logs/{i}/unified_{i}.png")

        model_response = explorer.get_response()

        print("Model response:", model_response)
        print("Expected label:", label)

        correct = check_validity_of_answer(model_response, label)
        total_answers += 1

        if correct:
            correct_answers += 1

        bar.set_postfix(accuracy=correct_answers / total_answers)

        responses = {
            "model_response": model_response,
            "expected_label": label,
            "correct": correct
        }

        with open(f"test_logs/{i}/responses.json", "w") as f:
            json.dump(responses, f, indent=4)

        sleep(20)  # This has to be done to avoid rate limiting

    print("Correct answers:", correct_answers)
    print("Total answers:", total_answers)
    print("Accuracy:", correct_answers / total_answers)


if __name__ == "__main__":
    main()
