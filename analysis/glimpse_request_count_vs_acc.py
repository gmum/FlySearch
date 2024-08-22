import os
import json

from matplotlib import pyplot as plt


def process_one_example(example_dir) -> tuple[bool, int] | None:
    with open(os.path.join(example_dir, "report.json")) as f:
        report = json.load(f)

        if report["model_response"] == "-1":
            return None

        if report["model_response"] == -1:
            raise ValueError("Model response is -1")

        correct = report["correct"]

        simplified_conversation = report["simplified_conversation"]
        user_messages = [mess[1] for mess in simplified_conversation if mess[0] == "user"]
        user_images = [mess["url"] for mess in user_messages if type(mess) is dict]

        number_glimpses = len(user_images)

        return correct, number_glimpses


def main():
    run_dir = "../all_logs/relative_position_gridline_5_glimpses_7"

    correct_per_glimpse = {i: 0 for i in range(1, 8)}
    total_per_glimpse = {i: 0 for i in range(1, 8)}
    failed_per_glimpse = {i: 0 for i in range(1, 8)}

    for example_dir in os.listdir(run_dir):
        full_dir = os.path.join(run_dir, example_dir)

        result = process_one_example(full_dir)

        if result is None:
            continue

        correct, number_glimpses = result

        total_per_glimpse[number_glimpses] += 1

        if correct:
            correct_per_glimpse[number_glimpses] += 1
        else:
            failed_per_glimpse[number_glimpses] += 1

    print("Total", total_per_glimpse)
    print("Correct", correct_per_glimpse)
    print("Failed", failed_per_glimpse)

    percentages = {}

    for key in total_per_glimpse:
        correct = 0

        if key in correct_per_glimpse:
            correct = correct_per_glimpse[key]

        try:
            percentages[key] = correct / total_per_glimpse[key]
        except:
            pass

    plt.bar(percentages.keys(), percentages.values())
    plt.xlabel("Number of glimpses")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs number of glimpses")
    # plt.show()

    # clear
    plt.clf()

    # Plot correct and incorrect as bars
    width = 0.3

    plt.bar(correct_per_glimpse.keys(), correct_per_glimpse.values(), label="Correct", width=width)
    plt.bar(list(map(lambda x: x + width, failed_per_glimpse.keys())), failed_per_glimpse.values(), label="Incorrect",
            width=width)
    plt.xlabel("Number of glimpses")
    plt.ylabel("Count")

    plt.legend()
    plt.title("Correct and failed counts vs number of glimpses")

    plt.show()


if __name__ == "__main__":
    main()
