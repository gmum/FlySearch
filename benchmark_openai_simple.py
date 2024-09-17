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

from abstract_conversation import Role
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

# subset = "direct_attributes"
subset = "relative_position"

RUN_NAME = f"{subset}_test_permutation_shift_NAIVE_1_FULL"

prompt_prefix = """You will be given a question and several answer options. You should choose the correct option based on the image provided to you. You just need to answer the question and do not need any information about individuals. When you are not sure about the answer, just guess the most likely one. To answer, simply copy entire text of one of the options in an XML <Answer> tag. Do not copy the letter meant to represent option's position.

 For example, if the options are:
    A. Option 1
    B. Option 2
    C. Option 3

 and you think the answer is "B. Option 2", respond with <Answer>Option 2</Answer>
 """


def main():
    ds = VstarSubBenchDataset(f"/home/dominik/vstar_bench/{subset}", transform=pil_to_opencv)
    # ds = torch.utils.data.Subset(ds, [0, 1])

    all_logs_dir = pathlib.Path("all_logs")
    run_dir = all_logs_dir / RUN_NAME

    all_logs_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    bar = tqdm(enumerate(ds), total=len(ds))  # type: ignore

    total_ok = 0
    total_examples = 0

    for i, (image, question, options, answer) in bar:
        psg = PermutationShiftGenerator(options)
        answer = answer.lower().strip()

        all_perms_dir = run_dir / str(i)
        all_perms_dir.mkdir(exist_ok=True)

        for perm_number, options_permutation in enumerate(psg):
            total_examples += 1
            conversation = OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY), model_name="gpt-4o")

            example_dir = all_perms_dir / str(perm_number)
            example_dir.mkdir(exist_ok=True)

            text = f"""{prompt_prefix}
            Question: {question}
            Options:
            {"\n".join(letter + ". " + option for letter, option in zip(string.ascii_uppercase, options_permutation))}

            """

            conversation.begin_transaction(role=Role.USER)
            conversation.add_text_message(text)
            conversation.add_image_message(opencv_to_pil(image))
            conversation.commit_transaction(send_to_vlm=True)

            model_response = str(conversation.get_latest_message()[1]).lower().strip()

            parser = XMLResponseParser()
            model_response = parser.get_answer(model_response) if parser.get_answer(model_response) else model_response

            # print("Answer:", answer)
            # print("Model response:", model_response)

            correct = model_response.endswith(answer)

            # print("Correct:", correct)

            if correct:
                total_ok += 1

            bar.set_postfix(accuracy=total_ok / total_examples)

            report = {
                "question": question,
                "options": options_permutation,
                "answer": answer,
                "model_response": model_response,
                "correct": correct,
                "conversation": conversation.get_conversation()
            }

            with open(example_dir / "report.json", "w") as f:
                json.dump(report, f, indent=4)

            cv2.imwrite(str(example_dir / "image.png"), image)


if __name__ == "__main__":
    main()
