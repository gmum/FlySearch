from lib2to3.pytree import convert

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
from abstract_response_parser import SimpleResponseParser
from xml_response_parser import XMLResponseParser
from prompt_generation import get_starting_prompt_for_vstar_explorer_xml, \
    get_classification_prompt_for_vstar_explorer_xml, get_starting_prompt_for_vstar_explorer, \
    get_classification_prompt_for_vstar_explorer
from intern_conversation import get_model_and_stuff, get_conversation, InternConversation
from permutation_shift_generator import PermutationShiftGenerator
from hr_bench import HRBench

subset = "hr_bench"
# subset = "relative_position"

RUN_NAME = f"{subset}_test_permutation_shift_INTRO_4"


def main():
    if subset == "hr_bench":
        ds = HRBench(transform=pil_to_opencv)
    else:
        ds = VstarSubBenchDataset(f"/home/dominik/vstar_bench/{subset}", transform=pil_to_opencv)

    ds = torch.utils.data.Subset(ds, [1])

    all_logs_dir = pathlib.Path("all_logs")
    run_dir = all_logs_dir / RUN_NAME

    all_logs_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    bar = tqdm(enumerate(ds), total=len(ds))  # type: ignore

    total_ok = 0
    total_examples = 0

    for i, (image, question, options, answer) in bar:
        psg = PermutationShiftGenerator(options)

        all_perms_dir = run_dir / str(i)
        all_perms_dir.mkdir(exist_ok=True)

        for perm_number, options_permutation in enumerate(psg):
            total_examples += 1
            conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY), model_name="gpt-4o")
            glimpse_generator = GridImageGlimpseGenerator(image, 5)

            example_dir = all_perms_dir / str(perm_number)
            example_dir.mkdir(exist_ok=True)

            explorer = VisualVStarExplorer(
                conversation=conversation,
                question=question,
                options=options_permutation,
                glimpse_generator=glimpse_generator,
                number_glimpses=7,
                response_parser=XMLResponseParser(),
                starting_prompt_generator=get_starting_prompt_for_vstar_explorer_xml,
                classification_prompt_generator=get_classification_prompt_for_vstar_explorer_xml,
            )

            model_response = None

            try:
                explorer.answer()
                model_response = str(explorer.get_response()).lower().strip()
            except:
                model_response = "ERROR"

            answer: str = answer.lower().strip()

            correct = model_response.endswith(answer)

            if correct:
                total_ok += 1

            bar.set_postfix(accuracy=total_ok / total_examples)

            report = {
                "question": question,
                "options": options_permutation,
                "model_response": model_response,
                "expected_response": answer,
                "correct": correct,
                "simplified_conversation": conversation.get_conversation(),
                # "conversation": conversation.get_entire_conversation(),
                "failed_coord_request": explorer.get_failed_coord_request(),
            }

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


if __name__ == "__main__":
    main()
