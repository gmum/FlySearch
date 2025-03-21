from argparse import ArgumentParser

from arg_resolvers import ScenarioArgResolver, ConversationFactoryResolver, LoggerFactoryResolver, RunnerResolver, Bus


def main():
    parser = ArgumentParser()

    bus = Bus([
        ScenarioArgResolver(),
        ConversationFactoryResolver(),
        LoggerFactoryResolver(),
        RunnerResolver(),
    ])

    bus.register_args(parser)
    args = parser.parse_args()
    config = bus.resolve_args(args, {})

    config["experiment_runner"].run()


if __name__ == "__main__":
    main()
