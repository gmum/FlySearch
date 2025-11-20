import os

from conversation.invalid_factory import InvalidFactoryFactory
from conversation.base_conversation_factory import BaseConversationFactory

try:
    from google import genai

    GEMINI_AVALIABLE = True
except ImportError:
    GEMINI_AVALIABLE = False


if GEMINI_AVALIABLE:
    from conversation.gemini_conversation import GeminiConversation

    class _GeminiFactory(BaseConversationFactory):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.client = genai.Client(api_key=os.environ["GEMINI_AI_KEY"])

        def get_conversation(self):
            return GeminiConversation(
                self.client,
                self.model_name,
                max_tokens=None,  # Avoid forcing max tokens; Gemini handles defaults
            )

    GeminiFactory = _GeminiFactory
else:
    GeminiFactory = InvalidFactoryFactory("gemini")
