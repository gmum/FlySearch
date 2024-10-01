import argparse
import pathlib

from openai import OpenAI

from conversation.openai_conversation import OpenAIConversation
from misc.config import OPEN_AI_KEY
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator, UnrealGridGlimpseGenerator
from prompts.drone_prompt_generation import generate_basic_drone_prompt
from explorers.drone_explorer import DroneExplorer


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


def get_conversation(args):
    if args.model == "gpt-4o":
        return OpenAIConversation(OpenAI(api_key=OPEN_AI_KEY), model_name="gpt-4o")


def get_prompt(args):
    if args.prompt == "basic":
        return generate_basic_drone_prompt


def perform_one_test(run_dir, prompt, glimpses, glimpse_generator, conversation, test_number):
    explorer = DroneExplorer(conversation, glimpse_generator, prompt, glimpses, (-50, -55, 100))
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

    for i in range(args.repeats):
        conversation = get_conversation(args)
        perform_one_test(run_dir, prompt, args.glimpses, generator, conversation, i)

    generator.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",
                        type=str,
                        required=True,
                        choices=["basic"]
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
                        choices=["gpt-4o"]
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

    args = parser.parse_args()

    run_dir = create_test_run_directory(args)
    repeat_test(args, run_dir)


if __name__ == "__main__":
    main()
