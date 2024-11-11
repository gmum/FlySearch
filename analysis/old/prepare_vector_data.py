import json
import os


def convert_examples(example_dir):
    total_ok = 0
    total_examples = 0

    for permutation_dir in os.listdir(example_dir):
        total_examples += 1
        json_path = os.path.join(example_dir, permutation_dir, "report.json")
        with open(json_path, "r") as f:
            report = json.load(f)
            correct = report["correct"]

            if correct:
                total_ok += 1

    return total_ok / total_examples


def get_mean_accuracies(base_run_dir) -> list[float]:
    accuracies = []
    for example_dir in os.listdir(base_run_dir):
        accuracies.append(convert_examples(os.path.join(base_run_dir, example_dir)))
    return accuracies


def main():
    base_run_dir = "../all_logs/relative_position_test_permutation_shift_FULL_1"
    accuracies = get_mean_accuracies(base_run_dir)

    for acc in accuracies:
        print(acc, end=" ")


if __name__ == "__main__":
    main()
