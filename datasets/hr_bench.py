import base64
import torch
import random
import pandas as pd

from PIL import Image
from io import BytesIO


class HRBench(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.data = pd.read_parquet("data/hr_bench_8k.parquet")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        raw_image = item["image"]
        image = Image.open(BytesIO(base64.b64decode(raw_image)))

        if self.transform:
            image = self.transform(image)

        answer = item["answer"]
        answer = item[answer]  # Yes.

        question = item["question"]

        options = [item["A"], item["B"], item["C"], item["D"]]

        return image, question, options, answer


def main():
    dataset = HRBench()
    print(f"Number of samples: {len(dataset)}")

    image, question, options, answer = dataset[0]
    print(f"Image: {image}")
    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
