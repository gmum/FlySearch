import base64
import cv2
import numpy as np

from enum import Enum
from typing_extensions import Buffer

from openai import OpenAI
from openai import RateLimitError
from time import sleep
from PIL import Image

from abstract_conversation import Conversation, Role


def pil_to_opencv(image: Image.Image) -> np.ndarray:
    return np.array(image)[:, :, ::-1].copy()


def opencv_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image[:, :, ::-1].copy())


class OpenAIConversation(Conversation):
    def __init__(self, client: OpenAI, model_name: str, seed=42, max_tokens=300):
        self.client = client
        self.conversation = []
        self.model_name = model_name
        self.seed = seed
        self.max_tokens = max_tokens

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

    def get_answer_from_openai(self, model, messages, max_tokens, seed):
        fail = True
        response = None

        while fail:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    seed=seed
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

        self.transaction_started = False
        self.transaction_role = None

        if not send_to_vlm:
            print("Not sending to VLM")
            return

        response = self.get_answer_from_openai(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=self.max_tokens,
            seed=self.seed
        )

        response_content = str(response.choices[0].message.content)
        response_role = Role.ASSISTANT

        self.begin_transaction(response_role)
        self.add_text_message(response_content)
        self.commit_transaction(send_to_vlm=False)

    def rollback_transaction(self):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation = {}

        self.transaction_started = False
        self.transaction_role = None

    def get_entire_conversation(self):
        return self.conversation

    def get_latest_message(self):
        return self.conversation[-1]


def main():
    from config import OPEN_AI_KEY
    from vstar_bench_dataset import VstarSubBenchDataset

    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(
        client,
        model_name="gpt-4o",
        seed=42,
        max_tokens=300
    )

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=pil_to_opencv)
    image, question, options, answer = ds[0]

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

    print(conversation.get_entire_conversation())


if __name__ == "__main__":
    main()
