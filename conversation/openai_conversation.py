import base64
import cv2

from time import sleep
from PIL import Image
from openai import RateLimitError, Client

from conversation.abstract_conversation import Conversation, Role
from misc.cv2_and_numpy import pil_to_opencv, opencv_to_pil


class OpenAIConversation(Conversation):
    def __init__(self, client: Client, model_name: str, seed=42, max_tokens=300, temperature=0.8, top_p=1.0):
        self.client = client
        self.conversation = []
        self.model_name = model_name
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.transaction_started = False
        self.transaction_role = None
        self.transaction_conversation = {}

    def begin_transaction(self, role: Role):
        if self.transaction_started:
            raise Exception("Transaction already started")

        self.transaction_started = True
        self.transaction_role = role

        role = "user" if role == Role.USER else "assistant"

        self.transaction_conversation = {
            "role": role,
            "content": []
        }

    def add_text_message(self, text: str):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        if self.transaction_conversation['role'] == 'assistant':
            self.transaction_conversation['content'] = text
        else:
            content = self.transaction_conversation["content"]
            content.append({
                "type": "text",
                "text": text
            })

    def add_image_message(self, image: Image.Image):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        image = image.convert("RGB")
        image = pil_to_opencv(image)
        base64_image = cv2.imencode('.jpeg', image)[1].tobytes()
        base64_image = base64.b64encode(base64_image).decode('utf-8')

        content = self.transaction_conversation["content"]

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"  # FIXME
                }
            }
        )

    def get_answer_from_openai(self, model, messages, max_tokens, seed, temperature, top_p):
        fail = True
        response = None

        while fail:
            try:
                response = self.client.chat.completions.create(
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
        return response

    def commit_transaction(self, send_to_vlm=False):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.conversation.append(self.transaction_conversation)
        self.transaction_conversation = {}

        if self.transaction_role == Role.ASSISTANT and send_to_vlm:
            raise Exception("Assistant cannot send messages to VLM")

        self.transaction_started = False
        self.transaction_role = None

        if not send_to_vlm:
            return

        response = self.get_answer_from_openai(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=self.max_tokens,
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p
        )

        response_content = str(response.choices[0].message.content)
        response_role = Role.ASSISTANT

        print("llm response:", response_content)

        self.begin_transaction(response_role)
        self.add_text_message(response_content)
        self.commit_transaction(send_to_vlm=False)

    def rollback_transaction(self):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation = {}

        self.transaction_started = False
        self.transaction_role = None

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
                        elif submessage["type"] == "image_url":
                            if save_urls:
                                yield role, submessage["image_url"]
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