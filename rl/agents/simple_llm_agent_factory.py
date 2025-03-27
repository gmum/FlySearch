from typing import Dict, Callable

from conversation import BaseConversationFactory
from navigators import GridDroneNavigator
from rl.agents import SimpleLLMAgent, BaseAgentFactory


class SimpleLLMAgentFactory(BaseAgentFactory):
    def __init__(self, conversation_factory: BaseConversationFactory):
        self.conversation_factory = conversation_factory

    def create_agent(self, prompt: str, **kwargs) -> SimpleLLMAgent:
        return SimpleLLMAgent(
            conversation=self.conversation_factory.get_conversation(),
            prompt=prompt,
            navigator=GridDroneNavigator()
        )
