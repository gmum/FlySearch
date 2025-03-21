from argparse import ArgumentParser, Namespace
from typing import List, Dict

from arg_resolvers import BaseArgResolver


class Bus(BaseArgResolver):
    def __init__(self, arg_resolvers: List[BaseArgResolver]):
        self.arg_resolvers = arg_resolvers

    def register_args(self, parser: ArgumentParser):
        for arg_resolver in self.arg_resolvers:
            arg_resolver.register_args(parser)

    def resolve_args(self, args: Namespace, accumulator: Dict) -> Dict:
        for arg_resolver in self.arg_resolvers:
            accumulator = arg_resolver.resolve_args(args, accumulator)

        return accumulator
