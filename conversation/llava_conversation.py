import typing

import torch

from transformers import pipeline
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModel
from transformers import BitsAndBytesConfig
from enum import Enum
from PIL import Image

from vstar_bench_dataset import VstarSubBenchDataset
from abstract_conversation import Conversation, Role
from cv2_and_numpy import pil_to_opencv, opencv_to_pil


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"


class LlavaConversation(Conversation):
    def __init__(self, client, seed=42):
        self.client = client
        self.conversation: list[tuple[Role, list[tuple[MessageType, str | Image.Image]]]] = []
        self.images: list[Image.Image] = []
        self.seed: int = seed

        self.transaction_started: bool = False
        self.transaction_role: Role | None = None
        self.transaction_conversation: list[tuple[MessageType, str | Image.Image]] = []

    @staticmethod
    def convert_conversation_to_llava_format(conversation):
        def convert_message(message: tuple[MessageType, str | Image.Image]):
            message_type, content = message

            if message_type == MessageType.TEXT:
                return content
            else:
                return "<image>"

        def iterate_over_messages():
            for role, messages in conversation:
                role_name = "USER" if role == Role.USER else "ASSISTANT"

                yield f"{role_name}: {'\n'.join([convert_message(message) for message in messages])}"

        return '\n'.join(list(iterate_over_messages()))

    def begin_transaction(self, role: Role):
        if self.transaction_started:
            raise Exception("Transaction already started")

        self.transaction_started = True
        self.transaction_role = role

    def add_text_message(self, text: str):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation.append((MessageType.TEXT, text))

    def add_image_message(self, image: Image.Image):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation.append((MessageType.IMAGE, image))
        self.images.append(image)

    def commit_transaction(self, send_to_vlm=False):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        if self.transaction_role == Role.ASSISTANT and send_to_vlm:
            raise Exception("Assistant cannot send messages to VLM")

        self.conversation.append((self.transaction_role, self.transaction_conversation))
        self.transaction_conversation = []
        self.transaction_started = False
        self.transaction_role = None

        if not send_to_vlm:
            return

        prompt = f'{self.convert_conversation_to_llava_format(self.conversation)}\nASSISTANT:'

        response = self.client(self.images, prompt)

        response_content = response.split("ASSISTANT:")[-1].strip().split("USER:")[0].strip()
        response_role = Role.ASSISTANT

        print("Adding: ", response_content)

        self.begin_transaction(response_role)
        self.add_text_message(response_content)
        self.commit_transaction(send_to_vlm=False)

    def rollback_transaction(self):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation = {}

        self.transaction_started = False
        self.transaction_role = None

    def _get_latest_message(self, conversation):
        latest = conversation[-1]

        speaker = latest[0]
        message = self.convert_conversation_to_llava_format([latest])
        delimiter = "USER:" if speaker == Role.USER else "ASSISTANT:"

        message = message.split(delimiter)[-1].strip()

        return speaker, message

    def get_latest_message(self):
        return self._get_latest_message(self.conversation)

    def get_conversation(self) -> typing.List[typing.Tuple[Role, str]]:

        def conversation_iterator():
            conversation = []

            for msg in self.conversation:
                conversation.append(msg)
                yield self._get_latest_message(conversation)

        return list(conversation_iterator())

    def __str__(self):
        return self.convert_conversation_to_llava_format(self.conversation)


class SimpleLlavaPipeline:
    def __init__(self, device="cpu", max_tokens=300):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                                   quantization_config=quantization_config)

        self.device = device
        self.max_tokens = max_tokens

    def __call__(self, images, prompt):
        inputs = self.processor(prompt, images, return_tensors='pt').to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens).to("cpu")
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return generated_text[0]


def main():
    import requests
    image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

    client = SimpleLlavaPipeline(device="cuda")

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=pil_to_opencv)
    image, question, options, answer = ds[0]
    image = opencv_to_pil(image)

    conversation = LlavaConversation(client)

    conversation.begin_transaction(Role.USER)
    conversation.add_image_message(image)
    conversation.add_image_message(image2)
    conversation.add_text_message("Are the places in UK?")
    conversation.commit_transaction(send_to_vlm=True)

    print("Latest")
    print(conversation.get_latest_message())

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("If that's not in UK, where is it?")
    conversation.commit_transaction(send_to_vlm=True)

    print("Latest")
    print(conversation.get_latest_message())

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("SAY SOMETHING SILLY, MAN!")
    conversation.commit_transaction(send_to_vlm=True)

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("Say a joke")
    conversation.commit_transaction(send_to_vlm=True)

    print("Entire conversation")
    print(conversation.get_conversation())


if __name__ == "__main__":
    main()
