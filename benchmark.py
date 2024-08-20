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
from visual_vstar_explorer import VisualVStarExplorer
from openai_conversation import OpenAIConversation
from cv2_and_numpy import pil_to_opencv, opencv_to_pil
from image_glimpse_generator import GridImageGlimpseGenerator
from visualization import ExplorationVisualizer

RUN_NAME = "run_1"


def main():
    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=pil_to_opencv)
    # ds = torch.utils.data.Subset(ds, range(60, len(ds)))

    all_logs_dir = pathlib.Path("all_logs")
    run_dir = all_logs_dir / RUN_NAME

    all_logs_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    bar = tqdm(enumerate(ds), total=len(ds))  # type: ignore

    total_ok = 0

    for i, (image, question, options, answer) in bar:
        conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY), model_name="gpt-4o")
        glimpse_generator = GridImageGlimpseGenerator(image, 5)

        explorer = VisualVStarExplorer(
            conversation=conversation,
            question=question,
            options=options,
            glimpse_generator=glimpse_generator,
            number_glimpses=5
        )

        model_response = ""

        try:
            explorer.answer()
            model_response = str(explorer.get_response()).lower().strip()
        except Exception as e:
            model_response = "ERROR DURING TESTING! " + str(e)

        answer = answer.lower().strip()

        correct = model_response == answer

        if correct:
            total_ok += 1

        bar.set_postfix(accuracy=total_ok / (i + 1))

        report = {
            "question": question,
            "options": options,
            "model_response": model_response,
            "expected_response": answer,
            "correct": correct,
            "simplified_conversation": conversation.get_conversation(),
            "conversation": conversation.get_entire_conversation(),
            "failed_coord_request": explorer.get_failed_coord_request(),
        }

        example_dir = run_dir / str(i)
        example_dir.mkdir(exist_ok=True)

        with open(example_dir / "report.json", "w") as f:
            json.dump(report, f, indent=4)

        cv2.imwrite(str(example_dir / "image.png"), image)

        try:
            visualizer = ExplorationVisualizer(
                glimpse_requests=explorer.get_glimpse_requests(),
                glimpse_generator=glimpse_generator,
            )

            visualizer.save_glimpse_boxes(str(example_dir / "glimpse_boxes.jpeg"))
            visualizer.save_glimpses_individually(str(example_dir / "glimpse"))
        except:
            pass

        break


if __name__ == "__main__":
    main()
