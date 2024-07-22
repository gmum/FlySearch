import base64

from enum import Enum
from typing_extensions import Buffer

from openai import OpenAI
from openai import RateLimitError
from time import sleep

from abstract_conversation import Message, Conversation


class OpenAITextMessage(Message):
    def __init__(self, text: str):
        self.text = text

    def payload(self) -> dict:
        return {
            "type": "text",
            "text": self.text
        }

    def __str__(self):
        return self.text


class OpenAIBase64ImageMessage(Message):

    def __init__(self, image: Buffer, image_type: str):
        self.base64 = base64.b64encode(image).decode('utf-8')
        self.image_type = image_type

    def payload(self) -> dict:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{self.image_type};base64,{self.base64}",
                "detail": "low"
            }
        }


class OpenAIConversation(Conversation):
    class OpenAIMessageAggregation(Message):
        """
        This class is meant to aggregate multiple messages and add info
        Whether it's a user or an assistant message.
        """

        def __init__(self, role: str, messages: list[Message]):
            self.messages = messages
            self.role = role

        def payload(self) -> dict:
            return {
                "role": self.role,
                "content": [message.payload() for message in self.messages]
            }

        def __str__(self):
            return "\n".join([str(message) for message in self.messages])

    def __init__(self, client: OpenAI):
        self.client = client
        self.conversation = []

    def get_answer_from_openai(self, model, messages, max_tokens):
        fail = True
        response = None

        while fail:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                )
                fail = False
            except RateLimitError as e:
                print("Rate limit error")
                print(e)
                sleep(120)
                fail = True

        return response

    def send_messages(self, *messages: Message) -> None:
        aggregate = OpenAIConversation.OpenAIMessageAggregation("user", list(messages))

        self.conversation.append(aggregate)

        payloads = [message.payload() for message in self.conversation]

        response = self.get_answer_from_openai(
            model="gpt-4o",
            messages=payloads,
            max_tokens=300
        )

        response_content = str(response.choices[0].message.content)
        response_role = "assistant"

        reply = OpenAIConversation.OpenAIMessageAggregation(
            response_role,
            [OpenAITextMessage(response_content)]
        )

        self.conversation.append(reply)

    def get_latest_message(self) -> Message:
        return self.conversation[-1]

    def get_entire_conversation(self) -> list[Message]:
        return self.conversation


def main():
    from config import OPEN_AI_KEY
    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(client)

    image = open("sample_images/burger.jpeg", "rb").read()

    conversation.send_messages(
        OpenAITextMessage("Hi, could you describe this image for me?"),
        OpenAIBase64ImageMessage(image, "jpeg")
    )

    print(conversation.get_latest_message())

    conversation.send_messages(
        OpenAITextMessage("What do you think of this image?")
    )

    print(conversation.get_latest_message())


if __name__ == "__main__":
    main()
