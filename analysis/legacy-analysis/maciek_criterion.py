import os
import pathlib
import statistics

from analysis.utils import iterate_over_end_locations_and_classes
from utils import iterate_over_experiment_coords_and_messages_time_series
from matplotlib import pyplot as plt


def is_success_criterion_satisfied(position: tuple[float, float, float], object_position: tuple[float, float, float],
                                   max_alt_diff=10) -> bool:
    higher_than_object = position[2] > object_position[2]

    alt_diff = position[2] - object_position[2]
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

    return higher_than_object and ok_alt_diff and object_within_view


def success_criterion_for_timeserie(timeserie, max_alt_diff=10, object_position=(0, 0, 9)):
    messages, coords = timeserie

    coord = coords[-1]

    return is_success_criterion_satisfied(coord, object_position, max_alt_diff=max_alt_diff)


def trim_time_serie(time_serie, trim_str="FOUND"):
    messages, coords = time_serie

    for i, msg in enumerate(messages):
        if trim_str in msg:
            return messages[:i + 1], coords[:i + 1]

    return messages, coords


def trim_str_in_time_serie(time_serie, trim_str="FOUND"):
    messages, coords = time_serie

    for msg in messages:
        if trim_str in msg:
            return True

    return False


def iterate_over_trimmed_time_series(root: pathlib.Path, trim_str="FOUND"):
    for time_serie in iterate_over_experiment_coords_and_messages_time_series(root, os.listdir(root)):
        yield trim_time_serie(time_serie, trim_str)


def get_only_found_time_series(time_series, trim_str="FOUND"):
    total = []

    for time_serie in time_series:
        if trim_str_in_time_serie(time_serie, trim_str=trim_str):
            total.append(time_serie)

    return total


def get_stats_for_time_series(time_series):
    claims = get_only_found_time_series(time_series, trim_str="FOUND")
    non_claims = [time_serie for time_serie in time_series if time_serie not in claims]

    total = len(time_series)

    success_criterion_for_claims = sum([success_criterion_for_timeserie(claim) for claim in claims])

    success_criterion_for_non_claims = sum([success_criterion_for_timeserie(non_claim) for non_claim in non_claims])

    return {
        "total": total,
        "claims": len(claims),
        "success_criterion_claim": success_criterion_for_claims,
        "non_claims": len(non_claims),
        "success_criterion_non_claim": success_criterion_for_non_claims,
        "claim_and_found_acc": success_criterion_for_claims / total,
        "accidental_finds": success_criterion_for_non_claims / total

    }


def aggregate_stats_for_time_series_per_start_position(time_series):
    start_positions = {}

    for time_serie in time_series:
        _, coords = time_serie

        start_position = coords[0]

        if start_position not in start_positions:
            start_positions[start_position] = []

        start_positions[start_position].append(time_serie)

    start_positions = {k: get_stats_for_time_series(v) for k, v in start_positions.items()}

    return start_positions


def get_stderr_from_binomial(acc, n):
    ok_examples = int(acc * n)
    not_ok_examples = n - ok_examples

    # The most gloriously inefficient way that could've been done analytically
    # At the same time, n = 5 (currently)
    arr = [1] * ok_examples + [0] * not_ok_examples

    return statistics.stdev(arr) / (n ** 0.5)


def plot_aggregated_time_series(aggregated_time_series: dict, ax, label):
    dict_list = sorted(list(aggregated_time_series.items()))

    accuracies = [v["claim_and_found_acc"] for k, v in dict_list]
    stderrs = [get_stderr_from_binomial(v["claim_and_found_acc"], v["total"]) for k, v in dict_list]

    starts = [z for (x, y, z), v in dict_list]

    bar = ax.errorbar(starts, accuracies, yerr=stderrs, marker="o", alpha=1, capsize=5, label=label)

    # add legend entry for bar

    ax.legend()

    return ax


def plot_time_series(time_series, ax, label):
    aggregated = aggregate_stats_for_time_series_per_start_position(time_series)

    return plot_aggregated_time_series(aggregated, ax, label)


def plot_basic_vs_adg():
    mc_0s_ct = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/MC-0S-CT")))
    mc_0s_cr = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/MC-0S-CR-2")))

    adg_0s_ct = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/ADG-MC-0S-CT")))
    adg_0s_cr = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/ADG-MC-0S-CR")))

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    ax[0].set_title("MC 0s")

    plot_time_series(mc_0s_ct, ax[0], label="MC 0s CT")
    plot_time_series(mc_0s_cr, ax[0], label="MC 0s CR")

    ax[1].set_title("ADG MC 0s")

    plot_time_series(adg_0s_ct, ax[1], label="ADG MC 0s CT")
    plot_time_series(adg_0s_cr, ax[1], label="ADG MC 0s CR")

    ax[0].set_xlabel("Start height")
    ax[1].set_xlabel("Start height")

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")

    plt.show()


def plot_success_criterion_per_class(ends_and_labels, ax):
    per_class_ends = {}

    for end, cls in ends_and_labels:
        success_criterion = is_success_criterion_satisfied(end, (0, 0, 0))

        if cls not in per_class_ends:
            per_class_ends[cls] = []

        per_class_ends[cls].append(success_criterion)

    averages = {k: sum(v) / len(v) for k, v in per_class_ends.items()}
    counts = {k: len(v) for k, v in per_class_ends.items()}
    stderrs = {k: get_stderr_from_binomial(averages[k], counts[k]) for k in per_class_ends}

    ax.bar(averages.keys(), averages.values(), yerr=stderrs.values(), capsize=5)


def main():
    # plot_basic_vs_adg()

    ends_and_labels = list(iterate_over_end_locations_and_classes(pathlib.Path("../all_logs/MC-0S-F"),
                                                                  os.listdir("../all_logs/MC-0S-F")))

    fig, ax = plt.subplots()

    plot_success_criterion_per_class(ends_and_labels, ax)
    plt.show()


if __name__ == "__main__":
    main()
