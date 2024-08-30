import numpy as np
import cv2
import os
import pathlib
import json
import torch
import string

from openai import OpenAI
from tqdm import tqdm

from vstar_bench_dataset import VstarSubBenchDataset
from cv2_and_numpy import pil_to_opencv, opencv_to_pil
from llava_conversation import LlavaConversation, SimpleLlavaPipeline
from abstract_conversation import Role
from intern_conversation import get_conversation, InternConversation, get_model_and_stuff
from xml_response_parser import XMLResponseParser


def eval_one_run(ds, model_params):
    prompt_prefix = """You will be given a question and several answer options. You should choose the correct option based on the image provided to you. You just need to answer the question and do not need any information about individuals. When you are not sure about the answer, just guess the most likely one. To answer, simply copy entire text of one of the options in an XML <Answer> tag. Do not copy the letter meant to represent option's position.

         For example, if the options are:
            A. Option 1
            B. Option 2
            C. Option 3

         and you think the answer is "B. Option 2", respond with <Answer>Option 2</Answer>
         """

    bar = tqdm(enumerate(ds), total=len(ds))
    bar.set_description(f"Detail level: {model_params['image_detail_level']}")

    total_ok = 0

    # pipeline = SimpleLlavaPipeline(device="cuda")

    for i, (image, question, options, answer) in bar:
        conversation = InternConversation(**model_params)
        answer = answer.lower().strip()
        text = f"""{prompt_prefix}
    Question: {question}
    Options:
    {"\n".join(letter + ". " + option for letter, option in zip(string.ascii_uppercase, options))}

    """

        image = opencv_to_pil(image)

        conversation.begin_transaction(role=Role.USER)
        conversation.add_text_message(text)
        conversation.add_image_message(image)
        conversation.commit_transaction(send_to_vlm=True)

        model_response = str(conversation.get_latest_message()[1]).lower().strip()

        parser = XMLResponseParser()
        model_response = parser.get_answer(model_response) if parser.get_answer(model_response) else model_response

        correct = model_response.endswith(answer)

        if correct:
            total_ok += 1

        bar.set_postfix(accuracy=total_ok / (i + 1))

    return total_ok / len(ds)


def main():
    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=pil_to_opencv)
    # ds = torch.utils.data.Subset(ds, range(1, 2))

    model_params = get_model_and_stuff()

    accuracies = []

    for detail_level in range(1, 50):
        # model_params['image_detail_level'] = detail_level

        acc = eval_one_run(ds, model_params)
        accuracies.append(acc)

    log_dir = pathlib.Path("all_logs")
    log_dir.mkdir(exist_ok=True)

    variance_testing = log_dir / "variance_testing"
    variance_testing.mkdir(exist_ok=True)

    run_name = "relative_position_detail_10"
    run_dir = variance_testing / run_name

    run_dir.mkdir(exist_ok=False)

    with open(run_dir / "accuracies.json", "w") as f:
        json.dump(accuracies, f, indent=4)


if __name__ == "__main__":
    main()
