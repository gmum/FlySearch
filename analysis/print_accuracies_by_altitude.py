import pathlib
import json

from matplotlib import pyplot as plt

from typing import List
from analysis import Run, RunAnalyser, CriterionPlotter, load_all_runs_from_a_dir

CITY_CATEGORIES = ['road_construction_site', 'crowd', 'large_trash_pile', 'fire', 'car']
FOREST_CATEGORIES = ['campsite', 'trash_pile', 'person', 'forest_fire', 'building']
ANOMALY_CATEGORY = ['anomaly']

ENVS = ["CityNew", "ForestNew"]
MODELS = ["GPT4o", "Sonnet", "Phi", "Qwen2-VL-72B", "Pixtral-Large-Instruct-2411", "llava-interleave-7b", "Sonnet",
          "llava-onevision", "InternVL", "Gemini"]

COLOURS = {'GPT4o': 'color0',  # CLOSED-SOURCE
           'Sonnet': 'color1',  # CLOSED-SOURCE
           'Gemini': 'color2',  # CLOSED-SOURCE
           'Phi': 'color3',  # OPEN-SOURCE, SMALL
           'InternVL': 'color4',  # OPEN-SOURCE, SMALL
           'llava-interleave-7b': 'color5',  # OPEN-SOURCE, SMALL
           'Qwen2-VL-72B': 'color6',  # OPEN-SOURCE, LARGE
           'llava-onevision': 'color7',  # OPEN-SOURCE, LARGE
           'Pixtral-Large-Instruct-2411': 'color8'}  # OPEN-SOURCE, LARGE

MODEL_NAME_TO_PAPER_NAME = {
    'GPT4o': 'GPT-4o',
    'Sonnet': 'Claude 3.5 Sonnet',
    'Gemini': 'Gemini 2.0 Flash',
    'Phi': 'Phi 3.5 vision',
    'InternVL': 'InternVL-2.5 8B MPO',
    'llava-interleave-7b': 'Llava-Interleave 7b',
    'Qwen2-VL-72B': 'Qwen2-VL 72B',
    'llava-onevision': 'Llava-Onevision 72b',
    'Pixtral-Large-Instruct-2411': 'Pixtral-Large 124B'

}


def group_runs(runs: List[Run], model_name: str):
    plotter = CriterionPlotter(runs)
    alt_aggregation_function = lambda run: (run.start_position[2] // 10) * 10
    no_anomaly_filter = lambda run: "anomaly" not in run.object_type.lower()

    runs_aggregated_per_altitude = plotter.aggregate_runs_per_function(fun=alt_aggregation_function,
                                                                       fil=no_anomaly_filter)

    fig, ax = plt.subplots(nrows=1)

    stats = plotter.plot_accuracy_in_aggregated_runs(runs_aggregated_per_altitude, ax=ax,
                                                     success_criterion=lambda x: x.model_claimed and RunAnalyser(
                                                         x).maciek_criterion_satisfied(10))

    print(rf"\addplot[color={COLOURS[model_name]}, mark=square]")
    print(r"coordinates {")

    for altitude, results in sorted(list(stats.items()), key=lambda x: int(x[0])):
        print(f"({altitude}, {results['mean'] * 100:.1f})")

    print(r"};")
    print(r"\addlegendentry{" + MODEL_NAME_TO_PAPER_NAME[model_name] + "}")


def main():
    path = pathlib.Path("../all_logs")

    for env in ENVS:
        if env != "ForestNew":
            continue
        for model in COLOURS.keys():
            runs = load_all_runs_from_a_dir(path / f"{model}-{env}")
            group_runs(runs, model)


if __name__ == "__main__":
    main()
