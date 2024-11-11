import json
import os
import pathlib
import shutil

from enum import Enum

class Result(Enum):
    REFUSAL_NO_OWL_FAIL = 1
    FAIL_OWL_FAIL = 2
    FAILED_PARSING = 3
    OK_OWL_OK = 4
    REFUSAL_OWL_FAIL = 5
    FAIL_NO_OWL_FAIL = 6
    OK_OWL_FAIL = 7

def process_report(report, example_dir):
    response = report["response"]
    expected = report["expected"]

    files_number = len(os.listdir(example_dir))

    if response == expected:
        if files_number == 3:
            return Result.OK_OWL_FAIL
        else:
            return Result.OK_OWL_OK
    elif "sorry" in response:
        if files_number == 3:
            return Result.REFUSAL_OWL_FAIL
        else:
            return Result.REFUSAL_NO_OWL_FAIL
    elif "options" in response:
        return Result.FAILED_PARSING
    else:
        if files_number == 3:
            return Result.FAIL_OWL_FAIL
        else:
            return Result.FAIL_NO_OWL_FAIL

def main():
    run_dir = "../all_logs/owl-full-norm-fix"
    run_dir = pathlib.Path(run_dir)

    script_output_dir = pathlib.Path("out/owl-full-norm-fix")
    script_output_dir.mkdir(exist_ok=False)

    results = {
        Result.REFUSAL_NO_OWL_FAIL: 0,
        Result.FAIL_OWL_FAIL: 0,
        Result.FAILED_PARSING: 0,
        Result.OK_OWL_OK: 0,
        Result.REFUSAL_OWL_FAIL: 0,
        Result.FAIL_NO_OWL_FAIL: 0,
        Result.OK_OWL_FAIL: 0,
    }

    for example in os.listdir(run_dir):
        example_dir = run_dir / example

        if not os.path.isdir(example_dir):
            continue

        with open(example_dir / "results.json") as f:
            report = json.load(f)
            result = process_report(report, example_dir)
            results[result] += 1

            case_output_dir = script_output_dir / str(result)
            case_output_dir.mkdir(exist_ok=True)

            shutil.copytree(example_dir, case_output_dir / example)


    print(results)

if __name__ == "__main__":
    main()