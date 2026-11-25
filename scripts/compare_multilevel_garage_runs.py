#!/usr/bin/env python3
"""Score a FLEX/TARE Garage pair from the unchanged common AEDE recorder files."""

import argparse
import bisect
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics


TARGET_PCT = 98.0
FLOOR_SPACING_M = 3.5
FLOOR_BAND_M = 0.45


@dataclass
class Run:
    path: Path
    method: str
    status: dict
    manifest: dict
    total_m3: float
    time_s: list
    volume_m3: list
    distance_m: list
    z_m: list


def finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: expected a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite number")
    return number


def load_json(path, optional=False):
    if optional and not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def agree(values, label):
    present = [finite_number(value, label) for value in values if value is not None]
    if any(not math.isclose(value, present[0], rel_tol=1e-9, abs_tol=1e-6)
           for value in present[1:]):
        raise ValueError(f"{label}: conflicting values {present}")
    return present[0] if present else None


def load_run(path, method):
    path = path.resolve()
    status = load_json(path / "status.json")
    if status.get("method") != method:
        raise ValueError(f"{path}: expected {method}, got {status.get('method')!r}")
    manifest = load_json(path.parent.parent / "campaign_manifest.json", optional=True)
    truth = manifest.get("truth_volume", {})
    total = agree((status.get("total_volume_m3"), manifest.get("total_volume_m3"),
                   truth.get("total_volume_m3")), f"{path}: reference volume")
    if total is None or total <= 0:
        raise ValueError(f"{path}: missing positive reference volume")
    agree((truth.get("total_volume_m3"),
           truth["unique_voxels"] * truth["voxel_size_m"] ** 3
           if "unique_voxels" in truth and "voxel_size_m" in truth else None),
          f"{path}: reference voxel volume")
    agree((status.get("duration_budget_s"), manifest.get("duration_s")),
          f"{path}: duration budget")

    fields = ("time_s", "explored_volume_m3", "traveling_distance_m")
    samples = [[], [], []]
    heights = []
    with (path / "common_metrics.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: common_metrics.csv lacks required columns {fields}")
        for line, row in enumerate(reader, start=2):
            values = [finite_number(row[field], f"{path}: CSV line {line} {field}")
                      for field in fields]
            if min(values) < 0:
                raise ValueError(f"{path}: negative cumulative value at CSV line {line}")
            if samples[0] and values[0] < samples[0][-1]:
                raise ValueError(f"{path}: reversed clock at CSV line {line}")
            for field, series, value in zip(fields[1:], samples[1:], values[1:]):
                if series and value + 1e-6 < series[-1]:
                    raise ValueError(f"{path}: decreasing {field} at CSV line {line}")
            if samples[0] and values[0] == samples[0][-1]:
                # A finish signal and the timed sample may share one clock tick.
                for series, value in zip(samples, values):
                    series[-1] = value
            else:
                for series, value in zip(samples, values):
                    series.append(value)
            if row.get("vehicle_z_m"):
                heights.append(finite_number(row["vehicle_z_m"], f"{path}: CSV line {line} z"))

    times, volumes, distances = samples
    if len(times) < 2 or times[-1] <= times[0] or volumes[-1] <= 0 or distances[-1] <= 0:
        raise ValueError(f"{path}: need advancing samples, positive volume and positive travel")
    recorder = status.get("recorder", {})
    if recorder.get("state") == "complete":
        agree((volumes[-1], recorder.get("last_explored_volume_m3")), f"{path}: final volume")
        agree((distances[-1], recorder.get("last_traveling_distance_m")), f"{path}: final distance")
    return Run(path, method, status, manifest, total, times, volumes, distances, heights)


def measure_at(x, y, at):
    if not x[0] <= at <= x[-1]:
        raise ValueError(f"{at} outside measured range [{x[0]}, {x[-1]}]")
    right = bisect.bisect_left(x, at)
    if x[right] == at:
        return y[right]
    left = right - 1
    fraction = (at - x[left]) / (x[right] - x[left])
    return y[left] + fraction * (y[right] - y[left])


def crossing(run, target_pct):
    target = target_pct * run.total_m3 / 100.0
    index = bisect.bisect_left(run.volume_m3, target - 1e-8)
    if index == len(run.volume_m3):
        return {"time_s": None, "distance_m": None, "volume_rate_m3_s": None,
                "right_censored_at_s": run.time_s[-1]}
    if index == 0:
        # The target was already observed when the recorded interval began.
        return {"time_s": None, "distance_m": None, "volume_rate_m3_s": None,
                "left_censored_at_s": run.time_s[0]}
    previous, current = run.volume_m3[index - 1], run.volume_m3[index]
    fraction = min(1.0, max(0.0, (target - previous) / (current - previous)))
    at = run.time_s[index - 1] + fraction * (run.time_s[index] - run.time_s[index - 1])
    return {"time_s": at, "distance_m": measure_at(run.time_s, run.distance_m, at),
            "volume_rate_m3_s": target / at if at > 0.0 else None}


def trajectory_heights(run):
    path = run.path / "aede_trajectory.txt"
    z_m = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line, text in enumerate(stream, start=1):
                if not text.strip():
                    continue
                fields = text.split()
                if len(fields) != 7:
                    raise ValueError(f"{path}: expected x y z roll pitch yaw time at line {line}")
                z_m.append(finite_number(fields[2], f"{path}: z at line {line}"))
    source = path.name if z_m else "common_metrics.csv"
    z_m = z_m or run.z_m
    if not z_m:
        return None
    start_z = statistics.median(z_m[:min(5, len(z_m))])
    floor_bands = {}
    for z in z_m:
        level = round((z - start_z) / FLOOR_SPACING_M)
        center_z = start_z + level * FLOOR_SPACING_M
        if abs(z - center_z) <= FLOOR_BAND_M:
            floor_bands[level] = floor_bands.get(level, 0) + 1
    return {"source": source, "sample_count": len(z_m), "start_z_m": start_z,
            "min_z_m": min(z_m), "max_z_m": max(z_m),
            "samples_near_nominal_deck_heights": dict(sorted(floor_bands.items()))}


def run_metrics(run, overlap_start, overlap_end, shared_target_pct):
    times, volumes, distances = run.time_s, run.volume_m3, run.distance_m
    volume_at_end = measure_at(times, volumes, overlap_end)
    area = 0.0
    prev_t = overlap_start
    prev_v = measure_at(times, volumes, overlap_start)
    for time in [t for t in times if overlap_start < t < overlap_end] + [overlap_end]:
        vol = measure_at(times, volumes, time)
        area += (vol + prev_v) * (time - prev_t) / 2.0
        prev_t, prev_v = time, vol
    last_gain = next(i for i in range(len(volumes) - 1, 0, -1)
                     if volumes[i] > volumes[i - 1]) if volumes[-1] > volumes[0] else 0
    return {
        "run_dir": str(run.path),
        "state": run.status.get("state"),
        "termination_reason": run.status.get("termination_reason"),
        "samples": len(times),
        "terminal_time_s": times[-1],
        "terminal_volume_m3": volumes[-1],
        "terminal_coverage_pct": 100.0 * volumes[-1] / run.total_m3,
        "terminal_distance_m": distances[-1],
        "terminal_avg_volume_rate_m3_s": volumes[-1] / times[-1],
        "thresholds": {f"t{pct}": crossing(run, pct) for pct in (80, 90, 95, 98)},
        "shared_target": crossing(run, shared_target_pct),
        "coverage_at_overlap_end_pct": 100.0 * volume_at_end / run.total_m3,
        "normalized_auc_over_measured_overlap_pct": (
            100.0 * area / ((overlap_end - overlap_start) * run.total_m3)),
        "last_volume_gain_sample_s": times[last_gain],
        "terminal_plateau_time_s": times[-1] - times[last_gain],
        "terminal_plateau_distance_m": distances[-1] - distances[last_gain],
        "trajectory_height_evidence": trajectory_heights(run),
    }


def score(flex, tare, horizon):
    total = agree((flex.total_m3, tare.total_m3), "paired reference volume")
    truths = [run.manifest.get("truth_volume", {}) for run in (flex, tare)]
    hashes = [truth.get("truth_ply_sha256") for truth in truths]
    if all(hashes) and hashes[0] != hashes[1]:
        raise ValueError("paired runs used different reference point-cloud hashes")
    voxel_size = agree([truth.get("voxel_size_m") for truth in truths], "paired voxel size")
    budgets = [agree((run.status.get("duration_budget_s"), run.manifest.get("duration_s")),
                     f"{run.method} duration budget") for run in (flex, tare)]
    budget = agree(budgets, "paired duration budget")
    commands = [run.status.get("substrate_command") for run in (flex, tare)]
    if all(commands) and commands[0] != commands[1]:
        raise ValueError("paired runs used different AEDE substrate launch commands")
    gate_provenance = all(hashes) and all(budgets) and all(commands) and voxel_size is not None
    gate_complete = all(run.status.get("state") == "completed" for run in (flex, tare))
    overlap_start = max(flex.time_s[0], tare.time_s[0])
    overlap_end = min(flex.time_s[-1], tare.time_s[-1])
    if horizon is not None:
        overlap_end = min(overlap_end, horizon)
    if overlap_end <= overlap_start:
        raise ValueError("the runs have no positive measured overlap")
    shared_target = min(TARGET_PCT, 100.0 * min(flex.volume_m3[-1], tare.volume_m3[-1]) / total)
    scores = {run.method: run_metrics(run, overlap_start, overlap_end, shared_target)
              for run in (flex, tare)}
    flex_t98 = scores["flex"]["thresholds"]["t98"]["time_s"]
    tare_t98 = scores["tare"]["thresholds"]["t98"]["time_s"]
    if not gate_provenance or not gate_complete:
        gate = "unverified_provenance_or_incomplete_run"
    elif scores["flex"]["terminal_coverage_pct"] < TARGET_PCT:
        gate = "flex_below_98_pct"
    elif flex_t98 is None or tare_t98 is None:
        gate = "t98_censored"
    elif flex_t98 >= tare_t98:
        gate = "flex_not_faster_at_98_pct"
    else:
        gate = "pass"
    return {
        "metric": "AEDE occupied-return voxel volume / fixed reference PLY voxel volume",
        "efficiency_gate": "FLEX reaches 98% and has lower interpolated T98 than matching TARE",
        "reference_volume_m3": total,
        "truth_ply_sha256_by_method": dict(zip(("flex", "tare"), hashes)),
        "voxel_size_m": voxel_size,
        "duration_budget_s": budget,
        "provenance_metadata_matches": bool(gate_provenance),
        "both_runs_completed_recording": gate_complete,
        "measured_overlap_start_s": overlap_start,
        "measured_overlap_end_s": overlap_end,
        "shared_attained_target_pct": shared_target,
        "t98_savings_s": tare_t98 - flex_t98 if flex_t98 is not None and tare_t98 is not None else None,
        "gate_status": gate,
        "runs": scores,
    }


def markdown(result):
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    rows = [
        "| Metric | FLEX | TARE |",
        "|---|---:|---:|",
    ]
    runs = [result["runs"][name] for name in ("flex", "tare")]
    for label, key in (
        ("Terminal coverage (%)", "terminal_coverage_pct"),
        ("Terminal time (s)", "terminal_time_s"),
        ("Terminal distance (m)", "terminal_distance_m"),
        ("Coverage at measured overlap end (%)", "coverage_at_overlap_end_pct"),
        ("Mean coverage over measured overlap (%)", "normalized_auc_over_measured_overlap_pct"),
        ("Distance after last gain (m)", "terminal_plateau_distance_m"),
        ("Time after last gain (s)", "terminal_plateau_time_s"),
    ):
        rows.append(f"| {label} | {fmt(runs[0][key])} | {fmt(runs[1][key])} |")
    for pct in (80, 90, 95, 98):
        values = [run["thresholds"][f"t{pct}"]["time_s"] for run in runs]
        rows.append(f"| T{pct} (s) | {fmt(values[0])} | {fmt(values[1])} |")
    for label, key in (("Travel to shared target (m)", "distance_m"),
                       ("Volume rate to shared target (m³/s)", "volume_rate_m3_s")):
        values = [run["shared_target"][key] for run in runs]
        rows.append(f"| {label} | {fmt(values[0])} | {fmt(values[1])} |")
    heights = [(run["trajectory_height_evidence"] or {}).get("max_z_m") for run in runs]
    rows.append(f"| Maximum vehicle z (m) | {fmt(heights[0])} | {fmt(heights[1])} |")
    rows += ["", f"Measured overlap {result['measured_overlap_start_s']:.2f} – "
             f"{result['measured_overlap_end_s']:.2f} s. "
             f"Shared attained target {result['shared_attained_target_pct']:.2f}%.",
             f"Comparison gate `{result['gate_status']}`.", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flex", type=Path, required=True, help="FLEX run directory")
    parser.add_argument("--tare", type=Path, required=True, help="TARE run directory")
    parser.add_argument("--horizon-s", type=float, help="upper bound on measured overlap")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="write to a new file instead of stdout")
    args = parser.parse_args()
    try:
        if args.horizon_s is not None and not (math.isfinite(args.horizon_s) and args.horizon_s > 0):
            raise ValueError("--horizon-s must be finite and positive")
        result = score(load_run(args.flex, "flex"), load_run(args.tare, "tare"), args.horizon_s)
        text = markdown(result) if args.format == "markdown" else json.dumps(
            result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Comparison error: {error}\n")


if __name__ == "__main__":
    main()
