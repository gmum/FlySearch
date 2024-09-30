import typing

import numpy as np
import torch
import torchvision

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from abstract_conversation import Conversation, Role
from llava_conversation import LlavaConversation

# Code from https://huggingface.co/OpenGVLab/InternVL2-8B

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        torchvision.transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_image_from_pil(image: Image.Image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternConversation(Conversation):
    def __init__(self, client, tokenizer, generation_config, image_detail_level=3, device="cuda"):
        self.client = client
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.history = None
        self.image_detail_level = image_detail_level
        self.device = device

        self.transaction_role: Role | None = None
        self.transaction_conversation: list[tuple[Role, str]] = []
        self.transaction_started: bool = False
        self.transaction_images: torch.tensor = []

    def begin_transaction(self, role: Role):
        if self.transaction_started is True:
            raise Exception("Transaction already started")

        if role == Role.ASSISTANT:
            raise NotImplementedError("Not implemented for assistant")

        self.transaction_role = role
        self.transaction_started = True
        self.transaction_conversation = []

    def add_text_message(self, text: str):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation.append((self.transaction_role, text))

    def add_image_message(self, image: Image.Image):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        image = load_image_from_pil(image, max_num=self.image_detail_level).to(torch.bfloat16).to(self.device)

        self.transaction_images.append(image)
        self.transaction_conversation.append((self.transaction_role, '<image>'))

    def commit_transaction(self, send_to_vlm=False):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        prompt = self._get_prompt_for_transaction(self.transaction_conversation)
        image_sizes = self._get_image_sizes()
        image_concatenated = self._get_image_concatenated()

        response, self.history = self.client.chat(
            self.tokenizer,
            image_concatenated,
            prompt,
            self.generation_config,
            num_patches_list=image_sizes,
            history=self.history,
            return_history=True
        )

        # print("=== GOT RESPONSE ===")
        # print(response)
        # print("=== END RESPONSE ===")

        self.transaction_started = False
        self.transaction_role = None
        self.transaction_conversation = []
        self.transaction_images = []

    def _get_image_sizes(self):
        return [image.size(0) for image in self.transaction_images]

    def _get_image_concatenated(self):
        if self.transaction_images == []:
            return None

        return torch.cat(self.transaction_images, dim=0)

    def _get_prompt_for_transaction(self, transaction_conversation):
        def convert_message(message: tuple[Role, str]):
            role, content = message
            return f"{content}"

        return '\n'.join([convert_message(message) for message in transaction_conversation])

    def get_conversation(self) -> typing.List[typing.Tuple[Role, str]]:

        def conversation_iterator(history):
            for i in range(0, len(history), 2):
                yield Role.USER, history[i]

                try:
                    yield Role.ASSISTANT, history[i + 1]
                except IndexError:
                    break

        total = []

        for subhistory in self.history:
            total.extend(list(conversation_iterator(subhistory)))

        return total

    def rollback_transaction(self):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_started = False
        self.transaction_role = None
        self.transaction_conversation = []
        self.transaction_images = []

    def get_latest_message(self) -> typing.Tuple[Role, str]:
        return self.get_conversation()[-1]

    def __str__(self):
        conv = self.get_conversation()
        return '\n'.join([f"{"User" if role == role.USER else "Assistant"}: {message}" for role, message in conv])


def main2():
    path = 'OpenGVLab/InternVL2-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    client = InternConversation(model, tokenizer, generation_config, image_detail_level=3, device="cuda")

    client.begin_transaction(Role.USER)
    client.add_image_message(Image.open('./sample_images/burger.jpeg'))
    client.add_text_message("Describe the image in detail.")
    client.commit_transaction(send_to_vlm=True)

    client.begin_transaction(Role.USER)
    client.add_text_message("Is this image humorous?")
    client.commit_transaction(send_to_vlm=True)

    print(client.get_conversation())
    print(client)


def get_conversation():
    stuff = get_model_and_stuff()

    conversation = InternConversation(**stuff)

    return conversation


def get_model_and_stuff():
    path = 'OpenGVLab/InternVL2-8B'

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval()

    # model = AutoModel.from_pretrained(
    #    path,
    #    torch_dtype=torch.bfloat16,
    #    low_cpu_mem_usage=True,
    #    use_flash_attn=False,
    #    trust_remote_code=True).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=512, do_sample=True)

    return {
        "client": model,
        "tokenizer": tokenizer,
        "generation_config": generation_config,
        "image_detail_level": 10,
        "device": "cuda"
    }


if __name__ == "__main__":
    main2()
