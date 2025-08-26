import os

from openai import OpenAI, _types

from conversation import BaseConversationFactory
from conversation.openai_conversation import OpenAIConversation


class GeminiFactory(BaseConversationFactory):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.environ["GEMINI_AI_KEY"], base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
                             max_retries=100)

        self.model_name = model_name

    def get_conversation(self):
        return OpenAIConversation(
            self.client,
            model_name=self.model_name,
            max_tokens=_types.NotGiven()
        ) # Without this, Gemini 2.5 Flash will give out Nones. Probably due to reasoning tokens. TODO: Pro still fails (sometimes (!)) for some reason.


def main():
    from PIL import Image
    from conversation import Role
    from dotenv import load_dotenv
    load_dotenv()


    image = Image.open("/home/dominik/Pobrane/burger.jpeg")

    factory = GeminiFactory("gemini-2.5-pro")
    conversation = factory.get_conversation()
    conversation.begin_transaction(Role.USER)

    conversation.add_text_message("Is this image humorous?")
    conversation.add_image_message(image)
    conversation.commit_transaction(send_to_vlm=False)

    conversation.begin_transaction(Role.ASSISTANT)
    conversation.add_text_message("No. It shows a picture of a cat.")
    conversation.commit_transaction(send_to_vlm=False)

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("Are you sure?")
    conversation.commit_transaction(send_to_vlm=True)

    print(conversation.get_latest_message()[1])

if __name__ == "__main__":
    main()
