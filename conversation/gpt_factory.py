import os

from openai import OpenAI, _types
from sympy.physics.units import temperature

from conversation.base_conversation_factory import BaseConversationFactory
from conversation.openai_conversation import OpenAIConversation


class GPTFactory(BaseConversationFactory):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])
        self.model_name = model_name.removeprefix("oai-")

    def get_conversation(self):
        return OpenAIConversation(
            self.client,
            model_name=self.model_name,
            max_tokens=_types.NotGiven(), # We have to do this because otherwise GPT-5 would stop working. 4o works with default arguments for this class, but while making this compatible with GPT-5 I've decided to stop passing these arguments altogether as they don't break the 4o.
            temperature=_types.NotGiven()
        )

def main():
    from PIL import Image
    from conversation import Role
    from dotenv import load_dotenv
    load_dotenv()


    image = Image.open("/home/dominik/Pobrane/burger.jpeg")

    factory = GPTFactory("oai-gpt-4o")
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