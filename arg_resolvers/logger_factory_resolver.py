import pathlib

from argparse import ArgumentParser, Namespace
from typing import Dict, List

from arg_resolvers import BaseArgResolver
from rl.evaluation.loggers import WandbLogger, WandbLoggerFactory, LocalFSLoggerFactory


class LoggerFactoryResolver(BaseArgResolver):
    def register_args(self, parser: ArgumentParser):
        parser.add_argument("--log_on_wandb", type=bool, required=False)
        parser.add_argument("--wandb_project_name", type=str, required=False)
        parser.add_argument("--log_directory", type=str, required=True)
        parser.add_argument("--run_name", type=str, required=True)

    def get_wandb_logger_factory(self, args: Namespace) -> List:
        if args.log_on_wandb:
            if args.wandb_project_name is None:
                raise ValueError("Wandb project name is required when logging on wandb")
            return [WandbLoggerFactory(args.wandb_project_name)]
        else:
            return []

    def get_local_logger_factory(self, args: Namespace) -> List:
        return [LocalFSLoggerFactory(pathlib.Path(args.log_directory) / args.run_name)]

    def resolve_args(self, args: Namespace, accumulator: Dict):

        accumulator["logger_factories"] = self.get_wandb_logger_factory(args) + self.get_local_logger_factory(args)

        return accumulator
