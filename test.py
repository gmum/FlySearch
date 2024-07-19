import torch
import torchvision
import cv2
import numpy as np

from openai import OpenAI

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

    conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY))
    subset = torch.utils.data.Subset(imagenet, range(1, 5000, 1000))

    for i, (image, label) in enumerate(subset):
        explorer = OpenAIVisualExplorer(conversation, image)
        explorer.classify()

        explorer.save_glimpse_boxes(f"test_logs/glimpses_{i}.png")
        explorer.save_glimpse_list(f"test_logs/glimpses_list_{i}.png")
        explorer.save_unified_image(f"test_logs/unified_{i}.png")

        model_response = explorer.get_response()

        print("Model response:", model_response)
        print("Expected label:", label)

        print("Correctness:", check_validity_of_answer(model_response, label))


if __name__ == "__main__":
    main()
