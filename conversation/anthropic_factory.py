import os

from conversation import AnthropicConversation
from anthropic import Anthropic



class AnthropicFactory:
    def __init__(self, model_name):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_AI_KEY"])

        model_name = model_name.removeprefix("anthropic-")
        self.model_name = model_name

    def get_conversation(self):
        return AnthropicConversation(
            self.client,
            model_name=self.model_name,
        )
