import torch
import json
import os

import torchvision.transforms
from PIL import Image


class VstarBenchDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):
        self.transform = transform
        self.data = []
        self.path = path
        jsonl_file = "test_questions.jsonl"
        with open(path + "/" + jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        test_obj = self.data[idx]

        img_rel_path = test_obj["image"]
        img_path = self.path + "/" + img_rel_path

        text = test_obj["text"]
        label = test_obj["label"]

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

            if self.transform:
                img = self.transform(img)

        return img, text, label


class VstarSubBenchDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.data = []
        self.path = path

        files = os.listdir(path)
        json_files = [f for f in files if f.endswith(".json")]

        images = [f for f in files if not f.endswith(".json")]
        image_name_to_extension = {f.split(".")[0]: f.split(".")[1] for f in images}

        self.json_files = json_files
        self.image_name_to_extension = image_name_to_extension

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        filename = self.json_files[idx]
        path = self.path + "/" + filename

        no_extension = filename.split(".")[0]
        image_filename = f"{no_extension}.{self.image_name_to_extension[no_extension]}"
        image_path = self.path + "/" + image_filename

        with open(path, "r") as metadata, open(image_path, "rb") as image:
            image = Image.open(image)

            if self.transform:
                image = self.transform(image)

            metadata = json.load(metadata)
            question = metadata["question"]
            options = metadata["options"]
            answer = options[0]

            return image, question, options, answer


def main():
    dataset = VstarBenchDataset("/home/dominik/vstar_bench")
    print(len(dataset))
    print(dataset[0])

    subsets = ["GPT4V-hard", "OCR", "relative_position", "direct_attributes"]

    subdatasets = [VstarSubBenchDataset(f"/home/dominik/vstar_bench/{s}", transform=torchvision.transforms.ToTensor())
                   for s in subsets]

    for s in subdatasets:
        print(len(s))


if __name__ == "__main__":
    main()
