from openai import OpenAI
from misc.config import OPEN_AI_KEY
from conversation.openai_conversation import OpenAIConversation


class GPTFactory:
    def __init__(self):
        self.client = OpenAI(api_key=OPEN_AI_KEY)

    def get_conversation(self, model_name):
        return OpenAIConversation(
            self.client,
            model_name=model_name,
        )
