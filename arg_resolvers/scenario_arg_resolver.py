import pathlib

from argparse import ArgumentParser, Namespace
from typing import Dict

from arg_resolvers import BaseArgResolver
from rl.environment import ForestFlySearchEnv, CityFlySearchEnv
from scenarios import MimicScenarioMapper, ForestScenarioMapper, DefaultForestScenarioMapper, BaseScenarioMapper, \
    CityScenarioMapper
from scenarios.default_city_scenario_mapper import DefaultCityScenarioMapper


class ScenarioArgResolver(BaseArgResolver):
    def register_args(self, parser: ArgumentParser):
        parser.add_argument("--scenario_type", type=str, choices=["forest_random", "city_random", "mimic"],
                            required=True)

        parser.add_argument(
            "--mimic_run_path", type=str, required=False
        )

        parser.add_argument(
            "--mimic_run_cls_names", type=str, required=False)

        parser.add_argument("--continue_from", type=int, required=False)

        parser.add_argument("--height_min", type=int, required=False)

        parser.add_argument("--height_max", type=int, required=False)

        parser.add_argument("--line_of_sight_assured", type=str, required=False, default="true")

        parser.add_argument("--alpha", type=float, required=False, default=0.5)

    def get_scenario_mapper(self, args: Namespace) -> BaseScenarioMapper:
        if args.scenario_type == "mimic":
            if args.height_min is not None or args.height_max is not None or args.alpha is not None:
                raise ValueError("Height min, max and alpha are not supported for mimic scenarios")

            return MimicScenarioMapper(pathlib.Path(args.mimic_run_path), args.mimic_run_cls_names, args.continue_from)

        if args.mimic_run_path is not None or args.mimic_run_cls_names is not None or args.continue_from is not None:
            raise ValueError("Mimic arguments are only supported for mimic scenarios")

        if args.scenario_type == "forest_random":
            return DefaultForestScenarioMapper(args.height_min, args.height_max, args.alpha)
        elif args.scenario_type == "city_random":
            return DefaultCityScenarioMapper(args.height_min, args.height_max, args.alpha)
        else:
            raise ValueError(f"Unknown scenario type: {args.scenario_type}")

    def get_environment(self, scenario_mapper: BaseScenarioMapper, args: Namespace) -> object:
        if scenario_mapper.get_object_type_cls() is ForestScenarioMapper.ObjectType:
            env = ForestFlySearchEnv()

            if not args.line_of_sight_assured:
                raise NotImplementedError("Line of sight is always assured for forest scenarios")

            return env
        elif scenario_mapper.get_object_type_cls() is CityScenarioMapper.ObjectType:
            env = CityFlySearchEnv()

            line_of_sight_assured = "true" in args.line_of_sight_assured.lower()

            env.set_throw_if_hard_config(line_of_sight_assured)

            return env
        else:
            raise ValueError(f"Unknown object type class: {scenario_mapper.get_object_type_cls()}")

    def resolve_args(self, args: Namespace, accumulator: Dict) -> Dict:
        scenario_mapper = self.get_scenario_mapper(args)
        environment = self.get_environment(scenario_mapper, args)

        accumulator["scenario_mapper"] = scenario_mapper
        accumulator["environment"] = environment

        return accumulator
