import json 
import torchvision
import random 

def get_dataset_length():
    imagenet = torchvision.datasets.ImageNet(
        root="/home/dominik/ImageNet",
        split="val",
    )

    return len(imagenet)

def main():
    length = get_dataset_length()
    cases = 50

    numbers = random.sample(range(length), cases)

    print(numbers)

    with open("subset_indexes.json", "w") as f:
        json.dump(numbers, f)

if __name__ == "__main__":
    main()