import json
import os


def convert_examples(example_dir) -> list[bool]:
    results = []

    for permutation_dir in os.listdir(example_dir):
        json_path = os.path.join(example_dir, permutation_dir, "report.json")
        with open(json_path, "r") as f:
            report = json.load(f)
            correct = report["correct"]

            results.append(correct)

    return results


def get_mean_accuracies(base_run_dir) -> list[bool]:
    results = []

    for example_dir in os.listdir(base_run_dir):
        results += convert_examples(os.path.join(base_run_dir, example_dir))

    return [1 if result else 0 for result in results]


def main():
    # base_run_dir = "../all_logs/relative_position_test_permutation_shift_FULL_1"
    # base_run_dir = "../all_logs/direct_attributes_test_permutation_shift_FULL_1"
    # base_run_dir = "../all_logs/relative_position_test_permutation_shift_NAIVE_1_FULL"
    base_run_dir = "../all_logs/direct_attributes_test_permutation_shift_NAIVE_1_FULL"

    accuracies = get_mean_accuracies(base_run_dir)

    for acc in accuracies:
        print(acc, end=" ")


if __name__ == "__main__":
    main()
