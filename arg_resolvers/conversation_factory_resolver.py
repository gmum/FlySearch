from argparse import ArgumentParser, Namespace
from typing import Dict

from arg_resolvers import BaseArgResolver
from conversation import GPTFactory, InternFactory, VLLMFactory
from conversation.anthropic_factory import AnthropicFactory
from conversation.gemini_factory import GeminiFactory


class ConversationFactoryResolver(BaseArgResolver):
    def register_args(self, parser: ArgumentParser):
        parser.add_argument("--model", type=str, required=True)

    def get_conversation_factory(self, args: Namespace):
        if args.model == "gpt-4o":
            return GPTFactory()
        elif args.model == "intern":
            return InternFactory()
        elif 'gemini' in args.model:
            return GeminiFactory(args.model)
        elif args.model.startswith('anthropic-'):
            return AnthropicFactory(args.model)
        else:
            return VLLMFactory(args.model)

    def resolve_args(self, args: Namespace, accumulator: Dict):
        accumulator["conversation_factory"] = self.get_conversation_factory(args)

        return accumulator
