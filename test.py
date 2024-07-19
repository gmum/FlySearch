import torch
import torchvision
import cv2
import numpy as np

from openai import OpenAI

from visual_explorer import OpenAIVisualExplorer
from abstract_conversation import Message, Conversation
from openai_conversation import OpenAIConversation
from config import OPEN_AI_KEY


def from_pil_to_opencv(image):
    return np.array(image)[:, :, ::-1].copy()


def main():
    imagenet = torchvision.datasets.ImageNet(
        root="/home/dominik/imagenet",
        split="val",
        transform=from_pil_to_opencv
    )

    conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY))
    subset = torch.utils.data.Subset(imagenet, range(5))

    for i, (image, label) in enumerate(subset):
        explorer = OpenAIVisualExplorer(conversation, image)
        explorer.classify()

        explorer.save_glimpse_boxes(f"test_logs/glimpses_{i}.png")
        explorer.save_glimpse_list(f"test_logs/glimpses_list_{i}.png")
        explorer.save_unified_image(f"test_logs/unified_{i}.png")


if __name__ == "__main__":
    main()
