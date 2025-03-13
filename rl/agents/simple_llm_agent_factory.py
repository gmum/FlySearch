from typing import Dict, Callable

from navigators import GridDroneNavigator
from rl.agents import SimpleLLMAgent, BaseAgentFactory


class SimpleLLMAgentFactory(BaseAgentFactory):
    def __init__(self, conversation_factory):
        self.conversation_factory = conversation_factory

    def create_agent(self, prompt: str) -> SimpleLLMAgent:
        return SimpleLLMAgent(
            conversation=self.conversation_factory.create_conversation(),
            prompt=prompt,
            navigator=GridDroneNavigator()
        )
