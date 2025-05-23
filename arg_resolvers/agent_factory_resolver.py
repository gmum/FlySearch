from arg_resolvers import BaseArgResolver

from rl.agents import SimpleLLMAgentFactory, DescriptionLLMAgentFactory, GeneralistOneAgentFactory


class AgentFactoryResolver(BaseArgResolver):
    def register_args(self, parser):
        parser.add_argument("--agent", type=str, required=True,
                            choices=["simple_llm", "description_llm", "generalist_one"], )

    def resolve_args(self, args, accumulator):
        if args.agent == "simple_llm":
            agent_factory = SimpleLLMAgentFactory(
                conversation_factory=accumulator["conversation_factory"],
            )
        elif args.agent == "description_llm":
            agent_factory = DescriptionLLMAgentFactory(
                conversation_factory=accumulator["conversation_factory"],
            )
        elif args.agent == "generalist_one":
            agent_factory = GeneralistOneAgentFactory(
                conversation_factory=accumulator["conversation_factory"],
            )
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")

        accumulator["agent_factory"] = agent_factory

        return accumulator
