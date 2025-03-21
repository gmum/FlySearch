from openai import OpenAI

from conversation.base_conversation_factory import BaseConversationFactory
from misc.config import OPEN_AI_KEY
from conversation.openai_conversation import OpenAIConversation


class GPTFactory(BaseConversationFactory):
    def __init__(self):
        self.client = OpenAI(api_key=OPEN_AI_KEY)

    def get_conversation(self):
        return OpenAIConversation(
            self.client,
            model_name="gpt-4o",
        )
