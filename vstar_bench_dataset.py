import torch
import json

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


def main():
    dataset = VstarBenchDataset("/home/dominik/vstar_bench")
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
