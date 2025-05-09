from argparse import ArgumentParser, Namespace
from typing import Dict

from arg_resolvers import BaseArgResolver
from prompts import fs1_prompt, fs2_prompt
from rl.agents import SimpleLLMAgentFactory
from rl.evaluation.configs import ExperimentConfig
from rl.evaluation.experiment_runner import ExperimentRunner
from rl.evaluation.validators import AltitudeValidatorFactory, OutOfBoundsFlightValidatorFactory, \
    RecklessFlyingValidatorFactory


class RunnerResolver(BaseArgResolver):
    def register_args(self, parser: ArgumentParser):
        parser.add_argument("--dummy_first", type=str, required=True)
        parser.add_argument("--forgiveness", type=int, required=True)
        parser.add_argument("--glimpses", type=int, required=True)
        parser.add_argument("--number_of_runs", type=int, required=True)
        parser.add_argument("--prompt_type", type=str, required=True, choices=["fs1", "fs2"])

    def resolve_args(self, args: Namespace, accumulator: Dict) -> Dict:
        conversation_factory = accumulator["conversation_factory"]
        environment = accumulator["environment"]
        mapper = accumulator["scenario_mapper"]

        fs1 = args.prompt_type == "fs1"
        prompt_factory = fs1_prompt if fs1 else fs2_prompt
        dummy_first = "true" in args.dummy_first.lower()

        additional_validators = []

        if fs1:
            additional_validators.append(RecklessFlyingValidatorFactory())

        config = ExperimentConfig(
            agent_factory=accumulator["agent_factory"],
            environment=environment,
            scenario_mapper=mapper,
            logger_factories=accumulator["logger_factories"],
            validator_factories=[AltitudeValidatorFactory(300),
                                 OutOfBoundsFlightValidatorFactory(fs2_behavior=not fs1)] + additional_validators,
            forgiveness=args.forgiveness,
            number_of_runs=args.number_of_runs,
            number_of_glimpses=args.glimpses,
            prompt_factory=prompt_factory
        )

        runner = ExperimentRunner(
            config=config,
            first_dummy=dummy_first
        )

        accumulator["experiment_runner"] = runner

        return accumulator
