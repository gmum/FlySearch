import os
import pathlib
from utils import iterate_over_experiment_coords_and_messages_time_series

def is_maciek_criterion_satisfied(position: tuple[float, float, float], object_position: tuple[float, float, float], max_alt_diff=10) -> bool:
    higher_than_object = position[2] > object_position[2]

    alt_diff = position[2] - object_position[2]
    ok_alt_diff = alt_diff <= max_alt_diff

    # Hello there, triangle similarity
    x_view_length = object_position[2] * (105 / 100)
    y_view_length = object_position[2] * (105 / 100)

    x_min_range = position[0] - x_view_length
    x_max_range = position[0] + x_view_length

    y_min_range = position[1] - y_view_length
    y_max_range = position[1] + y_view_length

    x_ok = x_min_range <= object_position[0] <= x_max_range
    y_ok = y_min_range <= object_position[1] <= y_max_range

    object_within_view = x_ok and y_ok

    return higher_than_object and ok_alt_diff and object_within_view

def maciek_criterion_for_timeserie(timeserie, max_alt_diff=10, object_position=(0, 0, 9)):
    messages, coords = timeserie

    coord = coords[-1]

    return is_maciek_criterion_satisfied(coord, object_position, max_alt_diff=max_alt_diff)

def trim_time_serie(time_serie, trim_str="FOUND"):
    messages, coords = time_serie

    for i, msg in enumerate(messages):
        if trim_str in msg:
            return messages[:i+1], coords[:i+1]

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

    maciek_criterion_for_claims = sum([maciek_criterion_for_timeserie(claim) for claim in claims])

    maciek_criterion_for_non_claims = sum([maciek_criterion_for_timeserie(non_claim) for non_claim in non_claims])

    return {
        "total": total,
        "claims": len(claims),
        "maciek_criterion_claim": maciek_criterion_for_claims,
        "non_claims": len(non_claims),
        "maciek_criterion_non_claim": maciek_criterion_for_non_claims
    }

def main():
    mc_1s_ct = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/maciek-criterion-incontext-center")))
    mc_0s_ct = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/maciek-criterion-nocontext-center")))
    mc_1s_cr = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/maciek-criterion-incontext-corner-2")))
    mc_0s_cr = list(iterate_over_trimmed_time_series(pathlib.Path("../all_logs/maciek-criterion-nocontext-corner")))

    for time_series in get_only_found_time_series(mc_1s_ct):
        print(time_series[0][-1])
        print("MACIEK CRITERION:", maciek_criterion_for_timeserie(time_series))

    print("MC 1s CT")
    print(get_stats_for_time_series(mc_1s_ct))

    print("MC 0s CT")
    print(get_stats_for_time_series(mc_0s_ct))

    print("MC 1s CR")
    print(get_stats_for_time_series(mc_1s_cr))

    print("MC 0s CR")
    print(get_stats_for_time_series(mc_0s_cr))

if __name__ == "__main__":
    main()
