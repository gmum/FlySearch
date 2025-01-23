from openai import OpenAI
from misc.config import GEMINI_AI_KEY
from conversation.openai_conversation import OpenAIConversation


class GeminiFactory:
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=GEMINI_AI_KEY, base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
                             max_retries=100)

        self.model_name = model_name

    def get_conversation(self):
        return OpenAIConversation(
            self.client,
            model_name=self.model_name,
        )


def main():
    from PIL import Image
    from conversation import Role

    image = Image.open("../data/sample_images/burger.jpeg")

    factory = GeminiFactory("gemini-2.0-flash-exp")
    conversation = factory.get_conversation()
    conversation.begin_transaction(Role.USER)

    conversation.add_text_message("Is this image humorous?")
    conversation.add_image_message(image)
    conversation.commit_transaction(send_to_vlm=True)

    print(conversation.get_latest_message()[1])

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("What is shown in this image?")
    conversation.commit_transaction(send_to_vlm=True)

    print(conversation.get_latest_message()[1])


if __name__ == "__main__":
    main()
