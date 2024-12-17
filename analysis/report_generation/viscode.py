import os
import sys
import json
import re
import pathlib

from typing import Tuple

if len(sys.argv) < 2:
    RUN_NAME = "MC-0S-CR"
    MODEL_NAME = "GPT-4o"
elif len(sys.argv) == 2:
    RUN_NAME = sys.argv[1]
    MODEL_NAME = "GPT-4o"
else:
    RUN_NAME = sys.argv[1]
    MODEL_NAME = sys.argv[2]

PATH_DIR = "../../all_logs/" + RUN_NAME


def boring_preamble():
    return """
    \\begin{drama}
    \\Character{Researcher}{re}
    \\Character{GPT-4o}{gpt}

    \\respeaks: [...]
    \\StageDir{\\re{} presents \\gpt{} with a \\(500 \\times 500\\) image of the starting location.}
    """.replace("GPT-4o", MODEL_NAME)


def total_preamble():
    return f"""
    \\documentclass[12pt]{'{article}'} 

    \\usepackage{'{notatki}'}
    \\fancyhead[L]{'{\\textsc{Scenario 1 / SCENARIONAME }}'}

    \\title{'{Scenario 1 / GPT-4o, SCENARIONAME }'}
    \\author{'{Dominik Matuszek}'}

    \\begin{'{document}'}

    \\maketitle
    """.replace("GPT-4o", MODEL_NAME).replace("SCENARIONAME", RUN_NAME)


def total_postamble():
    return f"""
    \end{'{document}'}
    """


def one_glimpse_latex(glimpse_path):
    return f"""
    \\begin{'{center}'}
    \includegraphics[scale=0.7]{'{'}{glimpse_path}{'}'}
    \\end{'{center}'}
    """


def gpt_speaks(speech):
    return f"\\gptspeaks: {speech}"


def stage_direction():
    return "\StageDir{\\re{} presents \\gpt{} with a relevant glimpse}"


def end_drama():
    return "\end{drama}"


def generate_section(name):
    return f"\\section{'{' + name + '}'}"


def generate_subsection(name):
    return f"\\subsection{'{' + name + '}'}"


def is_maciek_criterion_satisfied(position: tuple[float, float, float], object_position: tuple[float, float, float],
                                  max_alt_diff=10, object_highest_point=0) -> Tuple[bool, bool, bool]:
    higher_than_object = position[2] > object_position[2]

    alt_diff = position[2] - object_highest_point
    ok_alt_diff = alt_diff <= max_alt_diff

    # Hello there, triangle similarity
    x_view_length = (position[2] - object_position[2]) * (105 / 100)
    y_view_length = (position[2] - object_position[2]) * (105 / 100)

    x_min_range = position[0] - x_view_length
    x_max_range = position[0] + x_view_length

    y_min_range = position[1] - y_view_length
    y_max_range = position[1] + y_view_length

    x_ok = x_min_range <= object_position[0] <= x_max_range
    y_ok = y_min_range <= object_position[1] <= y_max_range

    object_within_view = x_ok and y_ok

    return higher_than_object, higher_than_object and object_within_view, higher_than_object and ok_alt_diff and object_within_view


def generate_metadata(starting_position, final_position, object_type="UNKNOWN", object_highest_point=0):
    higher_than_object, object_can_be_seen, maciek_criterion = is_maciek_criterion_satisfied(final_position, (
        0, 0, 0), object_highest_point=object_highest_point)

    return f"""
    \\begin{'{itemize}'}[h]
      \item Staring position: {starting_position}
      \item End position: {final_position} 
      \item Euclidean distance from (0, 0, 0): {(final_position[0] ** 2 + final_position[1] ** 2 + final_position[2] ** 2) ** 0.5}
      \item Higher than object: {higher_than_object}
      \item Object can be seen: {object_can_be_seen}
      \item Maciek criterion (with or without claim): {maciek_criterion}
      \item Object type: {object_type}
    \end{'{itemize}'}
    """


def str_to_tuple(tuple_str: str) -> tuple:
    ans = tuple_str.replace("(", "[")
    ans = ans.replace(")", "]")
    ans = tuple(json.loads(ans))

    return ans


def main():
    run_dir = pathlib.Path(PATH_DIR)

    print(total_preamble())

    for one_run_dir in sorted([subdir for subdir in os.listdir(run_dir)], key=lambda x: int(x.split("_")[0])):
        one_run_dir = run_dir / str(one_run_dir)

        print(generate_section(f"Example"))
        # print(generate_subsection("Basic information"))

        object_type = "UNKNOWN"

        try:
            with open(one_run_dir / "scenario_params.json") as f:
                json_obj = json.load(f)
                object_type = json_obj["object_type"]
        except OSError:
            pass

        try:
            with open(one_run_dir / "final_coords.txt") as f:
                final_coords = str_to_tuple(f.read())

            with open(one_run_dir / "start_rel_coords.txt") as f:
                start_coords = str_to_tuple(f.read())

            object_highest_point = 0

            try:
                with open(one_run_dir / "object_bbox.txt") as f:
                    object_highest_point = float(f.read().split()[5]) // 100
            except:
                pass

            print(generate_metadata(start_coords, final_coords, object_type, object_highest_point))
        except:
            pass

        print(boring_preamble())
        print(one_glimpse_latex(str(one_run_dir / "0.png")))

        i = 0

        while True:
            try:
                with open(one_run_dir / f"{i}.txt") as f:
                    speech = f.read()

                # with open(one_run_dir / f"{i}_coords.txt") as f:
                #    coords = f.read()

                print(gpt_speaks(speech))

            except OSError:
                break

            perhaps_image = one_run_dir / f"{i + 1}.png"
            if perhaps_image.exists():
                print(stage_direction())
                print(one_glimpse_latex(one_run_dir / f"{i + 1}.png"))
            else:
                break

            i += 1

        print(end_drama())

    print(total_postamble())


if __name__ == "__main__":
    main()
