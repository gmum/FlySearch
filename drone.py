import argparse
import pathlib

from openai import OpenAI

from conversation.openai_conversation import OpenAIConversation
from conversation.intern_conversation import InternConversation, get_model_and_stuff
from misc.config import OPEN_AI_KEY
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator, UnrealGridGlimpseGenerator
from prompts import generate_brute_force_drone_prompt
from prompts.drone_prompt_generation import generate_basic_drone_prompt, generate_xml_drone_prompt
from explorers.drone_explorer import DroneExplorer
from response_parsers.basic_drone_response_parser import BasicDroneResponseParser
from response_parsers.xml_drone_response_parser import XMLDroneResponseParser


def create_test_run_directory(args):
    all_logs_dir = pathlib.Path("all_logs")
    run_dir = all_logs_dir / args.run_name

    all_logs_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    return run_dir


def get_glimpse_generator(args):
    if args.glimpse_generator == "standard":
        return UnrealGlimpseGenerator()
    elif args.glimpse_generator == "grid":
        return UnrealGridGlimpseGenerator(splits_w=5, splits_h=5)


intern_model_and_stuff = None


def get_conversation(args):
    if args.model == "gpt-4o":
        return OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY), model_name="gpt-4o")
    elif args.model == "intern":
        global intern_model_and_stuff
        if intern_model_and_stuff is None:
            model_and_stuff = get_model_and_stuff()
        else:
            model_and_stuff = intern_model_and_stuff

        return InternConversation(**model_and_stuff)


def get_prompt(args):
    if args.prompt == "basic":
        return generate_basic_drone_prompt
    elif args.prompt == "brute_force":
        return generate_brute_force_drone_prompt
    elif args.prompt == "xml":
        return generate_xml_drone_prompt


def get_response_parser(args):
    if args.response_parser == "basic":
        return BasicDroneResponseParser()
    elif args.response_parser == "xml":
        return XMLDroneResponseParser()


def perform_one_test(run_dir, prompt, glimpses, glimpse_generator, conversation, response_parser, test_number):
    explorer = DroneExplorer(conversation, glimpse_generator, prompt, glimpses, (-50, -55, 100), response_parser)
    explorer.simulate()

    images = explorer.get_images()
    outputs = explorer.get_outputs()

    test_dir = run_dir / str(test_number)
    test_dir.mkdir(exist_ok=True)

    for i, (image, output) in enumerate(zip(images, outputs)):
        image.save(test_dir / f"{i}.png")
        with open(test_dir / f"{i}.txt", "w") as f:
            f.write(output)


def repeat_test(args, run_dir):
    generator = get_glimpse_generator(args)
    prompt = get_prompt(args)
    response_parser = get_response_parser(args)

    for i in range(args.repeats):
        try:
            conversation = get_conversation(args)
            perform_one_test(run_dir, prompt, args.glimpses, generator, conversation, response_parser, i)
        except:
            print(f"Failed on test {i}")
    generator.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",
                        type=str,
                        required=True,
                        choices=["basic", "brute_force", "xml"],
                        )

    parser.add_argument("--glimpses",
                        type=int,
                        required=True,
                        help="Number of glimpses to take. Note that the model can request number of glimpses - 1, as the first glimpse is always the starting glimpse."
                        )

    parser.add_argument("--glimpse_generator",
                        type=str,
                        required=True,
                        choices=["standard", "grid"],
                        help="Type of glimpse generator to use."
                        )

    parser.add_argument("--model",
                        type=str,
                        required=True,
                        choices=["gpt-4o", "intern"]
                        )

    parser.add_argument("--run_name",
                        type=str,
                        required=True,
                        help="Name of the run. This will be used to create a directory in the all_logs directory. If the directory already exists, the script will fail.")

    parser.add_argument("--repeats",
                        type=int,
                        required=True,
                        help="Number of times to repeat the test."
                        )

    parser.add_argument("--response_parser",
                        type=str,
                        required=True,
                        choices=["basic", "xml"]
                        )

    args = parser.parse_args()

    run_dir = create_test_run_directory(args)
    repeat_test(args, run_dir)


if __name__ == "__main__":
    main()
