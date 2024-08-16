import numpy as np
import cv2
import os
import pathlib
import json
import torch
import string

from openai import OpenAI
from tqdm import tqdm

from config import OPEN_AI_KEY
from vstar_bench_dataset import VstarSubBenchDataset
from openai_conversation import OpenAIConversation, OpenAITextMessage, OpenAIBase64ImageMessage
from visual_explorer_vstar import OpenAIVisualVStarExplorer


def from_pil_to_opencv(image):
    return np.array(image)[:, :, ::-1].copy()


def main():
    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=from_pil_to_opencv)
    # ds = torch.utils.data.Subset(ds, range(60, len(ds)))

    prompt_prefix = "You will be given a question and several answer options. You should choose the correct option based on the image provided to you. You just need to answer the question and do not need any information about individuals. When you are not sure about the answer, just guess the most likely one. To answer, simply copy entire text of one of the options. Do not copy the letter meant to represent option's position."

    pathlib.Path("test_logs").mkdir(exist_ok=True)
    bar = tqdm(enumerate(ds), total=len(ds))

    total_ok = 0

    for i, (image, question, options, answer) in bar:
        conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY))
        explorer = OpenAIVisualVStarExplorer(conversation, image, question, options)
        model_response = ""

        try:
            explorer.classify()
            model_response = str(explorer.get_response()).lower().strip()
        except Exception as e:
            model_response = "ERROR DURING TESTING! " + str(e)

        answer = answer.lower().strip()

        correct = model_response == answer

        if correct:
            total_ok += 1

        bar.set_postfix(accuracy=total_ok / (i + 1))

        report = {
            "conversation": str(conversation),
            "question": question,
            "options": options,
            "model_response": model_response,
            "expected_response": answer,
            "correct": correct,
            "conversation_list": [msg.payload() for msg in conversation.get_entire_conversation()]
        }

        pathlib.Path(f"test_logs/{i}").mkdir(exist_ok=True)
        with open(f"test_logs/{i}/report.json", "w") as f:
            json.dump(report, f, indent=4)

        cv2.imwrite(f"test_logs/{i}/image.jpeg", image)
        explorer.save_glimpses_individually(f"test_logs/{i}/glimpse")
        # explorer.save_unified_image(f"test_logs/{i}/unified_image.jpeg")

        try:
            explorer.save_glimpse_list(f"test_logs/{i}/glimpse_list.jpeg")
        except:
            pass
        explorer.save_glimpse_boxes(f"test_logs/{i}/glimpse_boxes.jpeg")


if __name__ == "__main__":
    main()
