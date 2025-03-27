from typing import Dict, Callable

from conversation import BaseConversationFactory
from navigators import GridDroneNavigator
from rl.agents import SimpleLLMAgent, BaseAgentFactory, DescriptionLLMAgent


class DescriptionLLMAgentFactory(BaseAgentFactory):
    def __init__(self, conversation_factory: BaseConversationFactory):
        self.conversation_factory = conversation_factory

    def create_agent(self, prompt: str, object_desc: str, **kwargs) -> SimpleLLMAgent:
        return DescriptionLLMAgent(
            conversation=self.conversation_factory.get_conversation(),
            prompt=prompt,
            navigator=GridDroneNavigator(),
            object_desc=object_desc,
            description_conversation_factory=self.conversation_factory
        )
