import os

from google import genai

from conversation.base_conversation_factory import BaseConversationFactory
from conversation.gemini_conversation import GeminiConversation


class GeminiFactory(BaseConversationFactory):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.environ["GEMINI_AI_KEY"])

    def get_conversation(self):
        return GeminiConversation(
            self.client,
            self.model_name,
            max_tokens=None  # Avoid forcing max tokens; Gemini handles defaults
        )
