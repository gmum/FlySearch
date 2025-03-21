from argparse import ArgumentParser, Namespace
from typing import Dict


class BaseArgResolver:
    def register_args(self, parser: ArgumentParser):
        raise NotImplementedError()

    def resolve_args(self, args: Namespace, accumulator: Dict) -> Dict:
        """
        :param args: Args from argparse.
        :param accumulator: Dict containing already initialized objects by other arg resolvers.
        :return: Dict containing initialized objects, initialized according to the args.
        """
        raise NotImplementedError()
