import argparse
import json
import pathlib
import traceback

from openai import OpenAI

from conversation import InternFactory, VLLMFactory
from conversation.gpt_factory import GPTFactory
from conversation.openai_conversation import OpenAIConversation
from conversation.intern_conversation import InternConversation, get_model_and_stuff
from misc.config import OPEN_AI_KEY
from glimpse_generators.unreal_glimpse_generator import UnrealGlimpseGenerator, UnrealGridGlimpseGenerator, \
    UnrealDescriptionGlimpseGenerator
from navigators import TrivialDroneNavigator, GridDroneNavigator
from prompts import generate_brute_force_drone_prompt, generate_xml_drone_grid_prompt, \
    generate_xml_drone_grid_prompt_with_grid_controls, xml_found_prompt, xml_found_prompt_cue
from prompts.drone_prompt_generation import generate_basic_drone_prompt, generate_xml_drone_prompt
from explorers.drone_explorer import DroneExplorer
from response_parsers.basic_drone_response_parser import BasicDroneResponseParser
from response_parsers.xml_drone_response_parser import XMLDroneResponseParser
from scenarios import DroneScenarioMapperWithOffsets, ScenarioConfigurator, CityScenarioMapper
from scenarios.drone_scenario_mapper import DroneScenarioMapper, YellowTruckScenarioMapper
from scenarios.forest_scenario_mapper import ForestScenarioMapper


def create_test_run_directory(args):
    if args.logdir:
        all_logs_dir = pathlib.Path(args.logdir)
    else:
        all_logs_dir = pathlib.Path("all_logs")
    run_dir = all_logs_dir / args.run_name

    all_logs_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    return run_dir


def get_glimpse_generator(args):
    if args.glimpse_generator == "standard":
        return UnrealGlimpseGenerator()
    elif args.glimpse_generator == "grid":
        return UnrealGridGlimpseGenerator(splits_w=6, splits_h=6)
    elif args.glimpse_generator == "description":
        return UnrealDescriptionGlimpseGenerator(
            conversation_factory=get_conversation_factory(args),
            searched_obj="yellow pickup truck",  # FIXME !!!
            splits_w=6,
            splits_h=6
        )


def get_conversation_factory(args):
    if args.model == "gpt-4o":
        return GPTFactory()
    elif args.model == "intern":
        return InternFactory()
    else:
        return VLLMFactory(args.model)


def get_prompt(args):
    if args.prompt == "basic":
        return generate_basic_drone_prompt
    elif args.prompt == "brute_force":
        return generate_brute_force_drone_prompt
    elif args.prompt == "xml":
        return generate_xml_drone_prompt
    elif args.prompt == "xml_grid":
        return generate_xml_drone_grid_prompt
    elif args.prompt == "xml_grid_grid":
        return generate_xml_drone_grid_prompt_with_grid_controls
    elif args.prompt == "xml_grid_grid_found":
        return xml_found_prompt
    elif args.prompt == "xml_found_prompt_cue":
        return xml_found_prompt_cue


def get_navigator(args):
    if args.navigator == "basic-bad":
        return TrivialDroneNavigator(BasicDroneResponseParser())
    elif args.navigator == "basic-xml":
        return TrivialDroneNavigator(XMLDroneResponseParser())
    elif args.navigator == "grid":
        return GridDroneNavigator()


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
    navigator = get_navigator(args)
    conversation_factory = get_conversation_factory(args)

    for i in range(args.repeats):
        try:
            conversation = conversation_factory.get_conversation()
            perform_one_test(run_dir, prompt, args.glimpses, generator, conversation, navigator, i)
        except:
            print(f"Failed on test {i}")
    generator.disconnect()


def get_scenario_mapper(args):
    if args.scenario_type == "level_1":
        return DroneScenarioMapperWithOffsets(
            min_x=args.x_offset_min,
            max_x=args.x_offset_max,
            step_x=args.x_offset_step,
            min_h=args.height_min,
            max_h=args.height_max,
            step_h=args.height_step,
            min_y=args.y_offset_min,
            max_y=args.y_offset_max,
            step_y=args.y_offset_step,
            scenario_mapper=DroneScenarioMapper()
        )
    elif args.scenario_type == "level_1_yellow_truck":
        return DroneScenarioMapperWithOffsets(
            min_x=args.x_offset_min,
            max_x=args.x_offset_max,
            step_x=args.x_offset_step,
            min_h=args.height_min,
            max_h=args.height_max,
            step_h=args.height_step,
            min_y=args.y_offset_min,
            max_y=args.y_offset_max,
            step_y=args.y_offset_step,
            scenario_mapper=YellowTruckScenarioMapper()
        )
    elif args.scenario_type == "forest":
        return ForestScenarioMapper(
            x_min=-15000,
            x_max=65000,
            y_min=-15000,
            y_max=65000,
            z_min=0,
            z_max=1,
            drone_z_rel_min=args.height_min * 100,
            drone_z_rel_max=args.height_max * 100,
            seed_min=1,
            seed_max=1000000000,
            scenarios_number=args.n,
            object_probs={
                (ForestScenarioMapper.ObjectType.HELICOPTER,
                 ForestScenarioMapper.ObjectType.PLANE,
                 ForestScenarioMapper.ObjectType.UFO): 0.0,
                (ForestScenarioMapper.ObjectType.PERSON,
                 ForestScenarioMapper.ObjectType.FIRE,
                 ForestScenarioMapper.ObjectType.TRASH,
                 ForestScenarioMapper.ObjectType.CAMPING,
                 ForestScenarioMapper.ObjectType.BUILDING): 1.0
            }
        )
    elif args.scenario_type == "city":
        return CityScenarioMapper(
            object_probs={
                (
                    CityScenarioMapper.ObjectType.POLICE_CAR,
                    CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR,
                    CityScenarioMapper.ObjectType.BLUE_SPORT_CAR,
                    CityScenarioMapper.ObjectType.RED_SPORT_CAR,
                    CityScenarioMapper.ObjectType.WHITE_SPORT_CAR,
                    CityScenarioMapper.ObjectType.CONSTRUCTION_WORKS,
                    CityScenarioMapper.ObjectType.FIRE,
                    CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK,
                    CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK,
                    CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK,
                    CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK,
                    CityScenarioMapper.ObjectType.CROWD,
                    CityScenarioMapper.ObjectType.TRASH,
                    CityScenarioMapper.ObjectType.WHITE_TRUCK,
                    CityScenarioMapper.ObjectType.BLACK_TRUCK
                ): 1.0
            },
            drone_z_rel_min=args.height_min * 100,
            drone_z_rel_max=args.height_max * 100,
            scenarios_number=args.n,
            seed_max=1000,
            seed_min=0,
        )


def scenario_level_test(args, run_dir):
    generator = get_glimpse_generator(args)
    gen_config = ScenarioConfigurator(generator)
    prompt = get_prompt(args)
    navigator = get_navigator(args)
    scenario_mapper = get_scenario_mapper(args)
    conversation_factory = get_conversation_factory(args)

    for i, scenario_dict in enumerate(scenario_mapper.iterate_scenarios()):
        for repeat in range(args.repeats):
            try:
                # Scenario configurator can alter scenario dict!!!
                gen_config.configure_scenario(scenario_dict, recovery_generator=scenario_mapper)
                drone_rel_coords = scenario_dict["drone_rel_coords"]
                object_type = scenario_dict["object_type"]
                object_id = scenario_dict[
                    "object_id"]  # Note: do not query this before configure scenario. It should set the object_id.
                object_bbox = gen_config.get_bbox(object_id)

                object_name = str(object_type.name)
                object_name = object_name.lower()
                object_name = object_name.replace("_", " ")
                scenario_dict["passed_object_name"] = object_name

                conversation = conversation_factory.get_conversation()
                explorer = DroneExplorer(
                    conversation=conversation,
                    glimpse_generator=generator,
                    prompt_generator=prompt,
                    glimpses=args.glimpses,
                    start_rel_position=drone_rel_coords,
                    navigator=navigator,
                    object_name=object_name,
                    incontext=(args.incontext == "True")
                )
                final_position = explorer.simulate()

                images = explorer.get_images()
                outputs = explorer.get_outputs()
                coordinates = explorer.get_coords()

                test_dir = run_dir / f"{str(i)}_r{str(repeat)} "
                test_dir.mkdir(exist_ok=True)

                with open(test_dir / "object_bbox.txt", "w") as f:
                    f.write(object_bbox)

                with open(test_dir / "scenario_params.json", "w") as f:
                    scenario_str_dict = {k: str(v) for k, v in scenario_dict.items()}
                    json.dump(scenario_str_dict, f, indent=4)

                for j, (image, output, location) in enumerate(zip(images, outputs, coordinates)):
                    image.save(test_dir / f"{j}.png")
                    with open(test_dir / f"{j}.txt", "w") as f:
                        f.write(output)
                    with open(test_dir / f"{j}_coords.txt", "w") as f:
                        f.write(str(location))

                with open(test_dir / "final_coords.txt", "w") as f:
                    f.write(str(final_position))

                with open(test_dir / "start_rel_coords.txt", "w") as f:
                    f.write(str(drone_rel_coords))

                with open(test_dir / "conversation.json", "w") as f:
                    if isinstance(conversation, OpenAIConversation):
                        conv = conversation.get_conversation(save_urls=False)
                    else:
                        conv = conversation.get_conversation()

                    conv = [(str(role), str(content)) for role, content in conv]
                    json.dump(conv, f, indent=4)

            except Exception as e:
                print(f"Failed on test {i}, repeat {repeat}", e)
                traceback.print_exc()

    generator.disconnect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--incontext",
                        type=str,
                        required=True,
                        choices=["True", "False"],
                        )

    parser.add_argument("--prompt",
                        type=str,
                        required=True,
                        choices=["basic", "brute_force", "xml", "xml_grid", "xml_grid_grid", "xml_grid_grid_found",
                                 "xml_found_prompt_cue"],
                        )

    parser.add_argument("--glimpses",
                        type=int,
                        required=True,
                        help="Number of glimpses to take. Note that the model can request number of glimpses - 1, as the first glimpse is always the starting glimpse."
                        )

    parser.add_argument("--glimpse_generator",
                        type=str,
                        required=True,
                        choices=["standard", "grid", "description"],
                        help="Type of glimpse generator to use."
                        )

    parser.add_argument("--model",
                        type=str,
                        required=True,
                        # choices=["gpt-4o", "intern"] # Now we have VLLM, so the name can be arbitrary
                        )

    parser.add_argument("--run_name",
                        type=str,
                        required=True,
                        help="Name of the run. This will be used to create a directory in the all_logs directory. If the directory already exists, the script will fail.")

    parser.add_argument("--scenario_type",
                        type=str,
                        required=True,
                        choices=["round_robin", "level_1", "level_1_yellow_truck", "forest", "city"],
                        )

    parser.add_argument("--repeats",
                        type=int,
                        required=True,
                        help="Number of times to repeat the test.",
                        default=1
                        )

    parser.add_argument("--navigator",
                        type=str,
                        required=True,
                        choices=["basic-bad", "basic-xml", "grid"]
                        )

    parser.add_argument("--height_min",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--height_max",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--height_step",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--x_offset_min",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--x_offset_max",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--x_offset_step",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--y_offset_min",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--y_offset_max",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--y_offset_step",
                        type=int,
                        required=False,
                        help="Only for use in scenarios.")

    parser.add_argument("--logdir",
                        type=str,
                        required=False,
                        help="Override for logs directory.")

    parser.add_argument("--n",
                        type=int,
                        required=False,
                        default=100,
                        help="Number of scenarios to generate. Doesn't do anything in round_robin mode.")

    args = parser.parse_args()

    run_dir = create_test_run_directory(args)

    if args.scenario_type == "round_robin":
        round_robin(args, run_dir)
    else:
        scenario_level_test(args, run_dir)


if __name__ == "__main__":
    main()
