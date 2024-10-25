import argparse
import pathlib

from openai import OpenAI

from conversation.openai_conversation import OpenAIConversation
from conversation.intern_conversation import InternConversation, get_model_and_stuff
from misc.config import OPEN_AI_KEY
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator, UnrealGridGlimpseGenerator
from prompts import generate_brute_force_drone_prompt, generate_xml_drone_grid_prompt
from prompts.drone_prompt_generation import generate_basic_drone_prompt, generate_xml_drone_prompt
from explorers.drone_explorer import DroneExplorer
from response_parsers.basic_drone_response_parser import BasicDroneResponseParser
from response_parsers.xml_drone_response_parser import XMLDroneResponseParser
from scenarios.drone_scenario_mapper import DroneScenarioMapper


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
    elif args.prompt == "xml_grid":
        return generate_xml_drone_grid_prompt


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
    coordinates = explorer.get_coords()

    test_dir = run_dir / str(test_number)
    test_dir.mkdir(exist_ok=True)

    for i, (image, output, location) in enumerate(zip(images, outputs, coordinates)):
        image.save(test_dir / f"{i}.png")
        with open(test_dir / f"{i}.txt", "w") as f:
            f.write(output)
        with open(test_dir / f"{i}_coords.txt", "w") as f:
            f.write(str(location))


def round_robin(args, run_dir):
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


def get_scenario_mapper(args):
    if args.scenario_type == "level_1":
        return DroneScenarioMapper()


def scenario_level_test(args, run_dir):
    generator = get_glimpse_generator(args)
    prompt = get_prompt(args)
    response_parser = get_response_parser(args)
    scenario_mapper = get_scenario_mapper(args)

    for i, (start_coords, object_name) in enumerate(scenario_mapper.iterate_scenarios()):
        try:
            generator.change_start_position(start_coords)
            conversation = get_conversation(args)
            explorer = DroneExplorer(
                conversation=conversation,
                glimpse_generator=generator,
                prompt_generator=prompt,
                glimpses=args.glimpses,
                start_rel_position=(0, 0, 120),
                response_parser=response_parser,
                object_name=object_name
            )
            final_position = explorer.simulate()

            images = explorer.get_images()
            outputs = explorer.get_outputs()
            coordinates = explorer.get_coords()

            test_dir = run_dir / str(i)
            test_dir.mkdir(exist_ok=True)

            for j, (image, output, location) in enumerate(zip(images, outputs, coordinates)):
                image.save(test_dir / f"{j}.png")
                with open(test_dir / f"{j}.txt", "w") as f:
                    f.write(output)
                with open(test_dir / f"{j}_coords.txt", "w") as f:
                    f.write(str(location))

            with open(test_dir / "final_coords.txt", "w") as f:
                f.write(str(final_position))

        except Exception as e:
            print(f"Failed on test {i}", e)

    generator.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",
                        type=str,
                        required=True,
                        choices=["basic", "brute_force", "xml", "xml_grid"],
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

    parser.add_argument("--scenario_type",
                        type=str,
                        required=True,
                        choices=["round_robin", "level_1"]
                        )

    parser.add_argument("--repeats",
                        type=int,
                        required=False,
                        help="Number of times to repeat the test. Only for round robin."
                        )

    parser.add_argument("--response_parser",
                        type=str,
                        required=True,
                        choices=["basic", "xml"]
                        )

    args = parser.parse_args()

    if args.scenario_type == "round_robin" and args.repeats is None:
        raise ValueError("Repeats must be specified for round robin scenario type.")

    if args.scenario_type != "round_robin" and args.repeats is not None:
        raise ValueError("Repeats should not be specified for scenario different than round robin.")

    run_dir = create_test_run_directory(args)

    if args.scenario_type == "round_robin":
        round_robin(args, run_dir)
    elif args.scenario_type == "level_1":
        scenario_level_test(args, run_dir)


if __name__ == "__main__":
    main()
