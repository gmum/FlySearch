import numpy as np
import cv2
import os
import pathlib
import json
import torch

from openai import OpenAI
from tqdm import tqdm

from config import OPEN_AI_KEY
from vstar_bench_dataset import VstarBenchDataset
from openai_conversation import OpenAIConversation, OpenAITextMessage, OpenAIBase64ImageMessage


def from_pil_to_opencv(image):
    return np.array(image)[:, :, ::-1].copy()


def main():
    ds = VstarBenchDataset("/home/dominik/vstar_bench", transform=from_pil_to_opencv)
    ds = torch.utils.data.Subset(ds, range(10))

    prompt_prefix = f"""You will be given a question and several answer options. You should choose the correct option based on the image provided to you. You just need to answer the question and do not need any information about individuals. When you are not sure about the answer, just guess the most likely one. WRITE ONLY THE LETTER, WITHOUT PARENTHESES.
"""

    pathlib.Path("test_logs").mkdir(exist_ok=True)
    bar = tqdm(enumerate(ds), total=len(ds))

    total_ok = 0

    for i, (image, text, label) in bar:
        conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY))
        text = prompt_prefix + text

        conversation.send_messages(
            OpenAITextMessage(text),
            OpenAIBase64ImageMessage(cv2.imencode('.jpeg', image)[1].tobytes(), "jpeg")
        )

        model_response = str(conversation.get_latest_message()).lower().strip()
        label = label.lower().strip()

        correct = model_response == label

        if correct:
            total_ok += 1

        bar.set_postfix(accuracy=total_ok / (i + 1))

        report = {
            "prompt": text,
            "model_response": model_response,
            "expected_response": label,
            "correct": correct
        }

        pathlib.Path(f"test_logs/{i}").mkdir(exist_ok=True)
        with open(f"test_logs/{i}/report.json", "w") as f:
            json.dump(report, f, indent=4)

        cv2.imwrite(f"test_logs/{i}/image.jpeg", image)


if __name__ == "__main__":
    main()
