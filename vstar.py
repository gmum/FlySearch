import argparse
import pathlib
import json
import torch
import cv2

from datasets.vstar_bench_dataset import VstarSubBenchDataset
from misc.cv2_and_numpy import pil_to_opencv, opencv_to_pil
from conversation.openai_conversation import OpenAIConversation
from openai import OpenAI
from misc.config import OPEN_AI_KEY
from explorers.vstar_owl_explorer import VstarOwlExplorer
from tqdm import tqdm

def save_test_params(args, run_path):
    with open(run_path / "test_params.json", "w") as f:
        json.dump(vars(args), f, indent=4)

def get_ds(args):
    if args.subset == "debug":
        ds = VstarSubBenchDataset("/home/dominik/vstar_bench/direct_attributes", transform=pil_to_opencv)
        ds = torch.utils.data.Subset(ds, range(3))
    elif args.subset == "attribute":
        ds = VstarSubBenchDataset("/home/dominik/vstar_bench/direct_attributes", transform=pil_to_opencv)
    elif args.subset == "spatial":
        ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=pil_to_opencv)
    else:
        raise NotImplementedError
    return ds

def get_conversation(args):
    if args.model == "gpt-4o":
        client = OpenAI(api_key=OPEN_AI_KEY)
        conversation = OpenAIConversation(
            client,
            model_name="gpt-4o",
        )
    else:
        raise NotImplementedError
    return conversation

def get_explorer(args, image, conversation, question, options):
    if args.explorer == "owl":
        explorer = VstarOwlExplorer(
            image=image,
            conversation=conversation,
            question=question,
            options=options
        )
    else:
        raise NotImplementedError
    return explorer

def save_results(result_dict, box_image, cutouts, example_path):
    with open(example_path / "results.json", "a") as f:
        json.dump(result_dict, f, indent=4)

    box_image.save(example_path / "box_image.jpg")
    for cutout_number, cutout in enumerate(cutouts):
        cutout.save(example_path / f"cutout_{cutout_number}.jpg")

def perform_run(args, run_path):
    ds = get_ds(args)
    bar = tqdm(enumerate(ds), total=len(ds))

    correct = 0

    for i, (image, question, options, answer) in bar:
        example_path = run_path / str(i)
        example_path.mkdir(exist_ok=False)
        cv2.imwrite(example_path / "image.jpg", image)

        conversation = get_conversation(args)
        explorer = get_explorer(args, image, conversation, question, options)

        try:
            response, box_image, cutouts, objects = explorer.get_answer()
        except Exception as e:
            with open(run_path / "errors.txt", "a") as f:
                f.write(f"Error in example {i}: {e}\n")
            continue

        response = response.lower()
        answer = answer.lower()

        if response == answer:
            correct += 1

        result_dict = {
            "correct": correct,
            "response": response,
            "expected": answer,
            "question": question,
            "options": options,
            "objects": objects,
            "conversation": conversation.get_conversation()
        }

        save_results(result_dict, box_image, cutouts, example_path)

        bar.set_postfix(accuracy=correct / (i + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explorer",
                        type=str,
                        required=True,
                        choices=["owl"],
                        )

    parser.add_argument("--model",
                        type=str,
                        required=True,
                        choices=["gpt-4o"],
                        )

    parser.add_argument("--experiment_name",
                        type=str,
                        required=True
                        )

    parser.add_argument("--subset",
                        type=str,
                        required=True,
                        choices=["debug", "attribute", "spatial"]
                        )

    args = parser.parse_args()

    run_path = pathlib.Path("all_logs") / args.experiment_name
    run_path.mkdir(exist_ok=False)

    save_test_params(args, run_path)
    perform_run(args, run_path)

if __name__ == "__main__":
    main()
