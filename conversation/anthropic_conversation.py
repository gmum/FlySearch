import base64
from time import sleep

import cv2
from PIL import Image
from anthropic import Anthropic, RateLimitError
from conversation import OpenAIConversation, Role
from misc.cv2_and_numpy import pil_to_opencv, opencv_to_pil


class AnthropicConversation(OpenAIConversation):
    def __init__(self, client: Anthropic, model_name: str, seed=42, max_tokens=300, temperature=0.8, top_p=1.0):
        super(AnthropicConversation, self).__init__(client, model_name, seed, max_tokens, temperature, top_p)

        self.post_transaction_image_counter = 0
        self.image_counter = 0

    def add_image_message(self, image: Image.Image):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.image_counter += 1
        self.add_text_message(f"Image {self.image_counter}:")

        image = image.convert("RGB")
        image = pil_to_opencv(image)
        base64_image = cv2.imencode('.jpeg', image)[1].tobytes()
        base64_image = base64.b64encode(base64_image).decode('utf-8')

        content = self.transaction_conversation["content"]

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": str(base64_image)
                }
            }
        )

    # FIXME: Renaming name of this function would be beneficial to clarity of this codebase, as we're definitely not calling OpenAI here
    def get_answer_from_openai(self, model, messages, max_tokens, seed, temperature, top_p):
        fail = True
        response = None

        while fail:
            try:
                response = self.client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    # seed=seed,
                    temperature=temperature,
                    top_p=top_p
                )
                fail = False
            except RateLimitError as e:
                print("Rate limit error")
                print(e)
                sleep(120)
                fail = True

        # FIXME: Dirty hack due to the fact that we are inheriting after OpenAIConversation
        # So we will make the return object look like the one from OpenAIConversation

        content = str(response.content[0].text)

        class Wrap:
            pass

        response = Wrap()
        setattr(response, "content", content)
        # .content

        message_wrap = Wrap()
        setattr(message_wrap, "message", response)
        # .message.content

        wrap = Wrap()
        setattr(wrap, "choices", [message_wrap])
        # wrap.choices[0].message.content

        return wrap

    def commit_transaction(self, send_to_vlm=False):
        super(AnthropicConversation, self).commit_transaction(send_to_vlm)
        self.post_transaction_image_counter = self.image_counter

    def rollback_transaction(self):
        super(AnthropicConversation, self).rollback_transaction()
        self.image_counter = self.post_transaction_image_counter

    def get_conversation(self, save_urls=True):
        def conversation_iterator():
            for message in self.conversation:
                role = Role.USER if message["role"] == "user" else Role.ASSISTANT
                content = message["content"]

                if isinstance(content, str):
                    yield role, content
                elif isinstance(content, list):
                    for submessage in content:
                        if submessage["type"] == "text":
                            yield role, submessage["text"]
                        elif submessage["type"] == "image":
                            if save_urls:
                                yield role, submessage["source"]["data"]
                            else:
                                yield role, "image"
                else:
                    raise Exception("Invalid content type")

        return list(conversation_iterator())

    def get_latest_message(self):
        if len(self.conversation) == 0:
            raise Exception("No messages in conversation")

        return self.get_conversation()[-1]

    def get_entire_conversation(self):
        return self.conversation


def main():
    from misc.config import ANTHROPIC_AI_KEY

    client = Anthropic(api_key=ANTHROPIC_AI_KEY)
    conversation = AnthropicConversation(
        client,
        model_name="claude-3-5-sonnet-20241022",
        seed=42,
        max_tokens=300,
        temperature=0.0000000000000000000001,
        top_p=0.0000000000000000000001
    )

    image = Image.open("/home/anonymous/MyStuff/active-visual-gpt/data/sample_images/burger.jpeg")
    image = pil_to_opencv(image)

    conversation.begin_transaction(Role.USER)
    conversation.add_image_message(opencv_to_pil(image))
    conversation.add_text_message("Hi, could you describe this image for me?")
    conversation.commit_transaction(send_to_vlm=False)

    conversation.begin_transaction(Role.ASSISTANT)
    conversation.add_text_message("This image depicts a goose.")
    conversation.commit_transaction(send_to_vlm=False)

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("Are you sure?")
    conversation.commit_transaction(send_to_vlm=True)

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("Is it humorous?")
    conversation.commit_transaction(send_to_vlm=True)

    print(conversation.get_conversation(save_urls=False))


if __name__ == "__main__":
    main()
