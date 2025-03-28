from argparse import ArgumentParser, Namespace
from typing import Dict

from arg_resolvers import BaseArgResolver
from prompts import xml_found_prompt
from rl.agents import SimpleLLMAgentFactory
from rl.evaluation.configs import ExperimentConfig
from rl.evaluation.experiment_runner import ExperimentRunner
from rl.evaluation.validators import AltitudeValidatorFactory, OutOfBoundsFlightValidatorFactory, \
    RecklessFlyingValidatorFactory


class RunnerResolver(BaseArgResolver):
    def register_args(self, parser: ArgumentParser):
        parser.add_argument("--dummy_first", type=bool, required=True)
        parser.add_argument("--forgiveness", type=int, required=True)
        parser.add_argument("--glimpses", type=int, required=True)
        parser.add_argument("--number_of_runs", type=int, required=True)

    def resolve_args(self, args: Namespace, accumulator: Dict) -> Dict:
        conversation_factory = accumulator["conversation_factory"]
        environment = accumulator["environment"]
        mapper = accumulator["scenario_mapper"]

        config = ExperimentConfig(
            agent_factory=accumulator["agent_factory"],
            environment=environment,
            scenario_mapper=mapper,
            logger_factories=accumulator["logger_factories"],
            validator_factories=[AltitudeValidatorFactory(300), OutOfBoundsFlightValidatorFactory(),
                                 RecklessFlyingValidatorFactory()],
            forgiveness=args.forgiveness,
            number_of_runs=args.number_of_runs,
            number_of_glimpses=args.glimpses,
            prompt_factory=xml_found_prompt
        )

        runner = ExperimentRunner(
            config=config,
            first_dummy=args.dummy_first
        )

        accumulator["experiment_runner"] = runner

        return accumulator
