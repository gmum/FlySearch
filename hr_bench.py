import base64
import torch
import random

from datasets import load_dataset
from PIL import Image
from io import BytesIO

ds = load_dataset("DreamMr/HR-Bench")


class HRBench(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.data = ds["train"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        raw_image = item["image"]
        image = Image.open(BytesIO(base64.b64decode(raw_image)))

        if self.transform:
            image = self.transform(image)

        answer = item["answer"]
        answer = item[answer]  # Yes.

        question = item["question"]

        options = [item["A"], item["B"], item["C"], item["D"]]
        random.shuffle(options)

        return image, question, options, answer


def main():
    ds = HRBench()

    for i in range(10):
        image, question, options, answer = ds[i]

        print(question)
        print(options)
        print(answer)

        image.show()

        break


if __name__ == "__main__":
    main()
