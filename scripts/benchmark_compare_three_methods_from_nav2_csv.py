#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy.spatial.distance import cdist


# Motion / evaluation constants used by plotting and metrics.
V_MAX = 0.22
W_MAX = 0.6
W_MIN = -0.6
VX_MAX = V_MAX
VX_MIN = -V_MAX
VY_MAX = V_MAX
VY_MIN = -V_MAX
DT = 0.033
GOAL_REACH_TOLERANCE_DIST_OMNI = 0.1
GOAL_REACH_TOLERANCE_HEADING = math.radians(0.3)


plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = True
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.handlelength"] = 1
mpl.rcParams["hatch.linewidth"] = 0.5


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    is_omni: bool
    color: str


@dataclass
class SimulationResult:
    poses: np.ndarray
    velocities_raw: np.ndarray
    ref_velocities_raw: np.ndarray
    break_flags: np.ndarray
    times: np.ndarray


METHOD_SPECS = [
    MethodSpec("dwpp", "DWPP for Diff-drive", False, "tab:blue"),
    MethodSpec("dwpp_omni_clip_min_l", "VP for Omni (min L)", True, "tab:green"),
    MethodSpec("dwpp_omni_clip_max_l", "VP for Omni (max L)", True, "tab:orange"),
    MethodSpec("dwpp_omni", "DWVP for Omni", True, "tab:red"),
]


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "results" / "compare_3_methods_from_nav2_csv"
DEFAULT_AUTO_PATH_ORDER = [
    "path3_right_angle_90_last_heading_minus_pi",
    "path4_one_minus_cos",
]
FILE_PREFIX_TO_METHOD_KEY = {
    "dwpp_nav2": "dwpp",
    "vpmin_nav2": "dwpp_omni_clip_min_l",
    "vpmax_nav2": "dwpp_omni_clip_max_l",
    "dwvp_nav2": "dwpp_omni",
}
AUTO_CSV_FILENAME_RE = re.compile(
    r"^(?P<prefix>[a-zA-Z0-9_]+)_(?P<date>\d{8}_\d{6})_(?P<nsec>\d{9})\.csv$"
)


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def append_heading_to_path(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty((0, 3))
    if len(path_xy) == 1:
        return np.array([[path_xy[0, 0], path_xy[0, 1], 0.0]])

    diffs = np.diff(path_xy, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])
    return np.c_[path_xy, headings]


def straight_line_heading_step_curve(segment_length: float = 1.0, points_per_segment: int = 100) -> np.ndarray:
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    headings_deg = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=float)
    headings_rad = np.deg2rad(headings_deg)
    n_segments = len(headings_rad)

    x = np.linspace(0.0, n_segments * segment_length, n_segments * points_per_segment + 1)
    y = np.zeros_like(x)

    theta = np.empty_like(x)
    for i, heading in enumerate(headings_rad):
        start = i * points_per_segment
        end = (i + 1) * points_per_segment
        theta[start:end] = heading
    theta[-1] = headings_rad[-1]

    return np.c_[x, y, theta]


def right_angle_polyline_curve(segment_length: float = 0.5, points_per_segment: int = 50) -> np.ndarray:
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    x1 = np.linspace(0.0, segment_length, points_per_segment + 1)
    y1 = np.zeros_like(x1)

    x2 = np.full(points_per_segment + 1, segment_length)
    y2 = np.linspace(0.0, segment_length, points_per_segment + 1)

    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    return append_heading_to_path(np.c_[x, y])


def right_angle_polyline_curve_last_segment_heading_minus_pi(
    segment_length: float = 0.5,
    points_per_segment: int = 50,
) -> np.ndarray:
    path = right_angle_polyline_curve(
        segment_length=segment_length,
        points_per_segment=points_per_segment,
    ).copy()
    path[-1, 2] = -np.pi
    return path


def _resample_xy_by_arclength(x: np.ndarray, y: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total_length = float(s[-1])

    if total_length <= 1e-12:
        return (
            np.linspace(float(x[0]), float(x[-1]), num_points),
            np.linspace(float(y[0]), float(y[-1]), num_points),
        )

    s_new = np.linspace(0.0, total_length, num_points)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new


def one_minus_cos_curve(
    amplitude: float = 1.0,
    length_x: float = 10.0,
    num_points: int = 200,
    cycles: float = 0.5,
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
    resample_arclength: bool = True,
) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if length_x <= 0.0:
        raise ValueError("length_x must be > 0")

    a = float(amplitude)
    l = float(length_x)
    k = 2.0 * math.pi * float(cycles) / l

    x = np.linspace(float(x0), float(x0) + l, num_points, dtype=float)
    u = x - float(x0)
    y = float(y0) + a * (1.0 - np.cos(k * u))

    if abs(theta0) > 0.0:
        c = math.cos(theta0)
        s = math.sin(theta0)
        x_shift = x - float(x0)
        y_shift = y - float(y0)
        x = float(x0) + c * x_shift - s * y_shift
        y = float(y0) + s * x_shift + c * y_shift

    if resample_arclength:
        x, y = _resample_xy_by_arclength(x, y, num_points)

    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)

    return np.c_[x, y, theta]


def parse_run_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run format (missing '='): {spec}")
    left, raw_path = spec.split("=", 1)
    if ":" not in left:
        raise ValueError(f"Invalid --run format (missing ':'): {spec}")
    path_name, method_key = left.split(":", 1)
    if not path_name or not method_key or not raw_path:
        raise ValueError(f"Invalid --run format: {spec}")
    return path_name, method_key, Path(raw_path)


def discover_run_specs_from_directory(
    input_dir: Path,
    path_order: list[str],
    use_latest_runs: int | None,
) -> list[str]:
    if not input_dir.exists():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")

    required_method_keys = [m.key for m in METHOD_SPECS]
    method_entries: dict[str, list[tuple[str, int, Path]]] = {k: [] for k in required_method_keys}

    for csv_path in input_dir.glob("*.csv"):
        m = AUTO_CSV_FILENAME_RE.match(csv_path.name)
        if m is None:
            continue
        prefix = m.group("prefix")
        method_key = FILE_PREFIX_TO_METHOD_KEY.get(prefix)
        if method_key not in method_entries:
            continue
        date_token = m.group("date")
        nsec = int(m.group("nsec"))
        method_entries[method_key].append((date_token, nsec, csv_path))

    for method_key in required_method_keys:
        method_entries[method_key].sort(key=lambda x: (x[0], x[1]))

    empty_methods = [k for k, v in method_entries.items() if len(v) == 0]
    if empty_methods:
        raise ValueError(
            "No matching CSV files found for methods: "
            f"{empty_methods}. "
            "Expected file prefixes: "
            f"{sorted(FILE_PREFIX_TO_METHOD_KEY.keys())}"
        )

    available_complete_runs = min(len(v) for v in method_entries.values())
    target_runs = use_latest_runs if use_latest_runs is not None else len(path_order)

    if target_runs <= 0:
        raise ValueError("--use-latest-runs must be > 0")
    if len(path_order) != target_runs:
        raise ValueError(
            "Number of --path-order entries must match number of runs to import. "
            f"path_order={len(path_order)}, target_runs={target_runs}"
        )
    if target_runs > available_complete_runs:
        counts = {k: len(v) for k, v in method_entries.items()}
        raise ValueError(
            "Requested runs exceed available complete sets. "
            f"requested={target_runs}, available={available_complete_runs}, counts={counts}"
        )

    selected_per_method: dict[str, list[Path]] = {}
    for method_key in required_method_keys:
        selected = method_entries[method_key][-target_runs:]
        selected_per_method[method_key] = [entry[2] for entry in selected]

    run_specs: list[str] = []
    print("[INFO] Auto-discovered CSV mapping:")
    for run_idx, path_name in enumerate(path_order):
        for method_key in required_method_keys:
            csv_path = selected_per_method[method_key][run_idx]
            print(f"  - {path_name}:{method_key}={csv_path}")
            run_specs.append(f"{path_name}:{method_key}={csv_path}")

    return run_specs


def _find_first_float(row: dict[str, str], names: list[str], default: float = np.nan) -> float:
    for n in names:
        if n not in row:
            continue
        v = row[n]
        if v is None or v == "":
            continue
        return float(v)
    return default


def _convert_poses_to_start_frame(poses: np.ndarray) -> np.ndarray:
    if len(poses) == 0:
        return poses

    valid_xy_idx = np.where(np.isfinite(poses[:, 0]) & np.isfinite(poses[:, 1]))[0]
    if len(valid_xy_idx) == 0:
        return poses

    idx0 = int(valid_xy_idx[0])
    x0 = float(poses[idx0, 0])
    y0 = float(poses[idx0, 1])
    yaw0 = float(poses[idx0, 2]) if np.isfinite(poses[idx0, 2]) else 0.0

    c = float(np.cos(yaw0))
    s = float(np.sin(yaw0))
    dx = poses[:, 0] - x0
    dy = poses[:, 1] - y0

    out = poses.copy()
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    out[:, 2] = np.where(np.isfinite(poses[:, 2]), normalize_angle(poses[:, 2] - yaw0), poses[:, 2])
    return out


def load_nav2_csv_result(csv_path: Path, trajectory_frame: str) -> SimulationResult:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    poses = []
    velocities_raw = []
    ref_velocities_raw = []
    break_flags = []
    times = []

    map_pose_valid_count = 0
    map_pose_total_rows = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_x = _find_first_float(row, ["x", "pose_x"])
            local_y = _find_first_float(row, ["y", "pose_y"])
            local_yaw = _find_first_float(row, ["yaw", "pose_yaw"])
            map_x = _find_first_float(row, ["map_x"], default=np.nan)
            map_y = _find_first_float(row, ["map_y"], default=np.nan)
            map_yaw = _find_first_float(row, ["map_yaw"], default=np.nan)
            map_pose_valid = _find_first_float(row, ["map_pose_valid"], default=np.nan)

            has_map_pose = np.isfinite(map_x) and np.isfinite(map_y) and np.isfinite(map_yaw)
            if np.isfinite(map_pose_valid):
                map_pose_total_rows += 1
                has_map_pose = has_map_pose and bool(int(round(map_pose_valid)))
                if has_map_pose:
                    map_pose_valid_count += 1

            x = map_x if has_map_pose else local_x
            y = map_y if has_map_pose else local_y
            yaw = map_yaw if has_map_pose else local_yaw

            vx_real = _find_first_float(row, ["vx_real", "speed_vx"], default=np.nan)
            vy_real = _find_first_float(row, ["vy_real", "speed_vy"], default=np.nan)
            w_real = _find_first_float(row, ["w_real", "speed_w"], default=np.nan)
            if not np.isfinite(vx_real):
                vx_real = _find_first_float(row, ["v_real"], default=0.0)
            if not np.isfinite(vy_real):
                vy_real = 0.0
            if not np.isfinite(w_real):
                w_real = _find_first_float(row, ["w_real", "w_cmd", "feedback_w"], default=0.0)

            vx_cmd = _find_first_float(row, ["vx_cmd", "cmd_vx", "desired_vx"], default=np.nan)
            vy_cmd = _find_first_float(row, ["vy_cmd", "cmd_vy", "desired_vy"], default=np.nan)
            w_cmd = _find_first_float(row, ["w_cmd", "cmd_w", "desired_w"], default=np.nan)
            if not np.isfinite(vx_cmd):
                vx_cmd = _find_first_float(row, ["v_cmd"], default=0.0)
            if not np.isfinite(vy_cmd):
                vy_cmd = 0.0
            if not np.isfinite(w_cmd):
                w_cmd = _find_first_float(row, ["w_cmd"], default=0.0)

            violation_val = _find_first_float(row, ["velocity_violation", "constraints_violation"], default=0.0)
            violation = bool(int(round(violation_val)))

            if "t" in row and row["t"] not in (None, ""):
                t = float(row["t"])
            else:
                sec = _find_first_float(row, ["sec"], default=0.0)
                nsec = _find_first_float(row, ["nsec"], default=0.0)
                t = sec + 1e-9 * nsec

            poses.append([x, y, yaw])
            velocities_raw.append([vx_real, vy_real, w_real])
            ref_velocities_raw.append([vx_cmd, vy_cmd, w_cmd])
            break_flags.append([violation, violation, violation])
            times.append(t)

    if len(poses) == 0:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    if map_pose_total_rows > 0:
        if map_pose_valid_count == 0:
            print(
                f"[WARN] map_pose_valid exists but no valid map pose rows in {csv_path}. "
                "Falling back to legacy x/y/yaw."
            )
        elif map_pose_valid_count < map_pose_total_rows:
            print(
                f"[INFO] Using map pose columns for {csv_path} "
                f"(valid rows: {map_pose_valid_count}/{map_pose_total_rows}; "
                "invalid rows fallback to legacy x/y/yaw)."
            )

    poses_np = np.array(poses, dtype=float)
    if trajectory_frame == "start":
        poses_np = _convert_poses_to_start_frame(poses_np)

    velocities_raw_np = np.array(velocities_raw, dtype=float)
    ref_velocities_raw_np = np.array(ref_velocities_raw, dtype=float)
    break_flags_np = np.array(break_flags, dtype=bool)
    times_np = np.array(times, dtype=float)
    times_np = times_np - times_np[0]

    return SimulationResult(
        poses=poses_np,
        velocities_raw=velocities_raw_np,
        ref_velocities_raw=ref_velocities_raw_np,
        break_flags=break_flags_np,
        times=times_np,
    )


def calc_path_headings(path: np.ndarray) -> np.ndarray:
    if path.shape[1] >= 3:
        return path[:, 2]
    if len(path) == 1:
        return np.array([0.0])
    diffs = np.diff(path[:, :2], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.concatenate([headings, [headings[-1]]])


def convert_velocity_to_vx_vy_w(spec: MethodSpec, velocities_raw: np.ndarray) -> np.ndarray:
    if spec.is_omni:
        return velocities_raw

    v = velocities_raw[:, 0]
    # diff-drive logs are loaded as [v_real, 0.0, w_real]
    w = velocities_raw[:, 2]
    out = np.zeros((len(velocities_raw), 3), dtype=float)
    out[:, 0] = v
    out[:, 2] = w
    return out


def calc_metrics(
    path: np.ndarray,
    result: SimulationResult,
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
) -> dict[str, float]:
    path_xy = path[:, :2]
    path_headings = calc_path_headings(path)
    robot_xy = result.poses[:, :2]
    robot_headings = result.poses[:, 2]

    distance_matrix = cdist(robot_xy, path_xy, metric="euclidean")
    nearest_indices = np.argmin(distance_matrix, axis=1)
    pos_errors = distance_matrix[np.arange(len(robot_xy)), nearest_indices]

    ref_headings = path_headings[nearest_indices]
    heading_errors = np.abs(normalize_angle(robot_headings - ref_headings))

    flags = result.break_flags[1:] if len(result.break_flags) > 1 else result.break_flags
    violation_rate = float(np.mean(np.any(flags, axis=1)) * 100.0)

    goal_distances = np.linalg.norm(robot_xy - path[-1, :2], axis=1)
    goal_heading_errors = np.abs(normalize_angle(robot_headings - path[-1, 2]))
    # Travel time is defined as CSV duration regardless of goal reach.
    if len(result.times) > 0:
        travel_time = float(result.times[-1] - result.times[0])
    else:
        travel_time = float("nan")

    return {
        "constraint_violation_rate_pct": violation_rate,
        "mean_position_error_m": float(np.mean(pos_errors)),
        "max_position_error_m": float(np.max(pos_errors)),
        "mean_heading_error_deg": float(np.rad2deg(np.mean(heading_errors))),
        "travel_time_s": travel_time,
    }


def format_value(value: float, digits: int = 4) -> str:
    if np.isfinite(value):
        return f"{value:.{digits}f}"
    return "N/A"


def write_metrics_tables(path_dir: Path, rows: list[dict[str, str | float]], figure_prefix: str = "") -> None:
    csv_path = path_dir / "metrics_table.csv"
    md_path = path_dir / "metrics_table.md"
    png_path = path_dir / f"{figure_prefix}metrics_table.png"

    headers = [
        "Method",
        "Constraint Violation Rate [%]",
        "Mean Position Error [m]",
        "Max Position Error [m]",
        "Mean Heading Error [deg]",
        "Travel Time [s]",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["Method"],
                format_value(float(row["constraint_violation_rate_pct"])),
                format_value(float(row["mean_position_error_m"])),
                format_value(float(row["max_position_error_m"])),
                format_value(float(row["mean_heading_error_deg"])),
                format_value(float(row["travel_time_s"])),
            ])

    with open(md_path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write(
                f"| {row['Method']} | "
                f"{format_value(float(row['constraint_violation_rate_pct']))} | "
                f"{format_value(float(row['mean_position_error_m']))} | "
                f"{format_value(float(row['max_position_error_m']))} | "
                f"{format_value(float(row['mean_heading_error_deg']))} | "
                f"{format_value(float(row['travel_time_s']))} |\n"
            )

    fig, ax = plt.subplots(figsize=(12, 1.8 + 0.5 * len(rows)))
    ax.axis("off")
    cell_text = []
    for row in rows:
        cell_text.append([
            row["Method"],
            format_value(float(row["constraint_violation_rate_pct"])),
            format_value(float(row["mean_position_error_m"])),
            format_value(float(row["max_position_error_m"])),
            format_value(float(row["mean_heading_error_deg"])),
            format_value(float(row["travel_time_s"])),
        ])
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def draw_path_heading_arrows(ax: plt.Axes, path: np.ndarray, color: str = "black") -> None:
    headings = calc_path_headings(path)
    n = len(path)
    if n == 0:
        return
    step = max(1, n // 24)
    indices = np.arange(0, n, step, dtype=int)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = max(0.24, 0.040 * diag)

    x = path[indices, 0]
    y = path[indices, 1]
    dx = arrow_length * np.cos(headings[indices])
    dy = arrow_length * np.sin(headings[indices])

    ax.quiver(
        x,
        y,
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        alpha=1.00,
        width=0.006,
        headwidth=5.0,
        headlength=6.0,
        headaxislength=5.0,
        label="Path Heading",
    )


def set_equal_axis_with_min_span(
    ax: plt.Axes,
    xy_arrays: list[np.ndarray],
    min_span: float = 1.0,
    margin_ratio: float = 0.1,
) -> None:
    all_xy = np.vstack(xy_arrays)
    x_min = float(np.min(all_xy[:, 0]))
    x_max = float(np.max(all_xy[:, 0]))
    y_min = float(np.min(all_xy[:, 1]))
    y_max = float(np.max(all_xy[:, 1]))

    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    x_half = 0.5 * x_span * (1.0 + margin_ratio)
    y_half = 0.5 * y_span * (1.0 + margin_ratio)

    ax.set_xlim(x_center - x_half, x_center + x_half)
    ax.set_ylim(y_center - y_half, y_center + y_half)
    ax.set_aspect("equal")


def calc_tracking_layout(
    xy_arrays: list[np.ndarray],
    min_span_default: float = 1.0,
) -> tuple[tuple[float, float], float, float]:
    all_xy = np.vstack(xy_arrays)
    x_span_raw = float(np.max(all_xy[:, 0]) - np.min(all_xy[:, 0]))
    y_span_raw = float(np.max(all_xy[:, 1]) - np.min(all_xy[:, 1]))

    min_span_tight = max(0.20, 0.12 * max(x_span_raw, 1e-6))
    min_span = min(min_span_default, min_span_tight) if x_span_raw > y_span_raw else min_span_default

    x_span = max(x_span_raw, min_span)
    y_span = max(y_span_raw, min_span)
    data_aspect = x_span / max(y_span, 1e-6)

    if data_aspect >= 1.6:
        fig_height = 3.0
        fig_width = min(10.5, max(7.0, fig_height * data_aspect))
        margin_ratio = 0.08
    else:
        fig_width = 3.0
        fig_height = 3.0
        margin_ratio = 0.12

    return (fig_width, fig_height), min_span, margin_ratio


def save_tracking_plots_by_method(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path,
    file_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    pose_arrow_length = max(0.06, 0.055 * diag)
    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    fig_size, min_span, margin_ratio = calc_tracking_layout(xy_arrays, min_span_default=1.0)

    for spec in method_specs:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
        draw_path_heading_arrows(ax, path)

        poses = results[spec.key].poses
        ax.plot(poses[:, 0], poses[:, 1], color=spec.color, linewidth=1.0, label=spec.label)

        step = max(1, len(poses) // 25)
        indices = np.arange(0, len(poses), step, dtype=int)
        if indices[-1] != len(poses) - 1:
            indices = np.append(indices, len(poses) - 1)

        dx = pose_arrow_length * np.cos(poses[indices, 2])
        dy = pose_arrow_length * np.sin(poses[indices, 2])
        ax.quiver(
            poses[indices, 0],
            poses[indices, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=spec.color,
            alpha=1.00,
            width=0.0055,
            headwidth=4.5,
            headlength=5.5,
            headaxislength=4.5,
        )

        set_equal_axis_with_min_span(ax, xy_arrays, min_span=min_span, margin_ratio=margin_ratio)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}tracking_poses_{spec.key}.png", dpi=200)
        plt.close(fig)


def save_tracking_plot_overlaid(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_path: Path,
) -> None:
    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    fig_size, min_span, margin_ratio = calc_tracking_layout(xy_arrays, min_span_default=1.0)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
    draw_path_heading_arrows(ax, path)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    pose_arrow_length = max(0.06, 0.055 * diag)
    for spec in method_specs:
        poses = results[spec.key].poses
        ax.plot(poses[:, 0], poses[:, 1], color=spec.color, linewidth=1.0, label=spec.label)

        step = max(1, len(poses) // 25)
        indices = np.arange(0, len(poses), step, dtype=int)
        if indices[-1] != len(poses) - 1:
            indices = np.append(indices, len(poses) - 1)

        dx = pose_arrow_length * np.cos(poses[indices, 2])
        dy = pose_arrow_length * np.sin(poses[indices, 2])
        ax.quiver(
            poses[indices, 0],
            poses[indices, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=spec.color,
            alpha=1.00,
            width=0.0055,
            headwidth=4.5,
            headlength=5.5,
            headaxislength=4.5,
        )

    set_equal_axis_with_min_span(ax, xy_arrays, min_span=min_span, margin_ratio=margin_ratio)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_velocity_profiles_by_method(
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path,
    file_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_cache = {}
    time_max = 0.0
    for spec in method_specs:
        result = results[spec.key]
        v_real = convert_velocity_to_vx_vy_w(spec, result.velocities_raw)
        v_ref = convert_velocity_to_vx_vy_w(spec, result.ref_velocities_raw)
        converted_cache[spec.key] = (result, v_real, v_ref)
        if len(result.times) > 0:
            time_max = max(time_max, float(result.times[-1]))

    values_min_v = [VX_MIN, VY_MIN]
    values_max_v = [VX_MAX, VY_MAX]
    for spec in method_specs:
        _, v_real, v_ref = converted_cache[spec.key]
        values_min_v.extend(
            [
                float(np.min(v_real[:, 0])),
                float(np.min(v_ref[:, 0])),
                float(np.min(v_real[:, 1])),
                float(np.min(v_ref[:, 1])),
            ]
        )
        values_max_v.extend(
            [
                float(np.max(v_real[:, 0])),
                float(np.max(v_ref[:, 0])),
                float(np.max(v_real[:, 1])),
                float(np.max(v_ref[:, 1])),
            ]
        )
    y_min_v = min(values_min_v)
    y_max_v = max(values_max_v)
    span_v = max(y_max_v - y_min_v, 1e-6)
    y_lim_v = (y_min_v - 0.08 * span_v, y_max_v + 0.08 * span_v)

    omega_span = max(W_MAX - W_MIN, 1e-6)
    omega_margin = 0.08 * omega_span
    y_lim_w = (W_MIN - omega_margin, W_MAX + omega_margin)

    for spec in method_specs:
        result, v_real, v_ref = converted_cache[spec.key]

        fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.8), sharex=True)

        axes[0].plot(result.times, v_ref[:, 0], color="red", linewidth=1.5)
        axes[0].plot(result.times, v_real[:, 0], color="blue", linewidth=1.5)
        axes[0].plot(result.times, v_ref[:, 1], color="red", linewidth=1.5, linestyle="--")
        axes[0].plot(result.times, v_real[:, 1], color="blue", linewidth=1.5, linestyle="--")
        axes[0].axhline(VX_MAX, color="black", linestyle="--", linewidth=0.8)
        if abs(VY_MAX - VX_MAX) > 1e-9:
            axes[0].axhline(VY_MAX, color="0.35", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel(r"$v_x, v_y$ [m/s]")
        axes[0].set_xlim(0.0, time_max if time_max > 0.0 else 1.0)
        axes[0].set_ylim(*y_lim_v)
        axes[0].grid(True)

        axes[1].plot(result.times, v_ref[:, 2], color="red", linewidth=1.5)
        axes[1].plot(result.times, v_real[:, 2], color="blue", linewidth=1.5)
        axes[1].axhline(W_MAX, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel(r"$\omega$ [rad/s]")
        axes[1].set_xlim(0.0, time_max if time_max > 0.0 else 1.0)
        axes[1].set_ylim(*y_lim_w)
        axes[1].grid(True)
        axes[1].set_xlabel("Time [s]")
        plt.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}velocity_profiles_{spec.key}.png", dpi=200)
        fig.savefig(output_dir / f"{file_prefix}velocity_profiles_{spec.key}.pdf", bbox_inches="tight")
        plt.close(fig)


def save_tracking_animation(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    set_equal_axis_with_min_span(ax, xy_arrays, min_span=1.0, margin_ratio=0.2)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("Path Tracking Animation")
    ax.grid(True)

    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
    draw_path_heading_arrows(ax, path)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = max(0.05, 0.05 * diag)

    trail_lines = {}
    pose_points = {}
    pose_arrows = {}
    for spec in method_specs:
        trail_line, = ax.plot([], [], color=spec.color, linewidth=1.1, label=spec.label)
        pose_point, = ax.plot([], [], marker="o", color=spec.color, markersize=5)
        pose_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=10, color=spec.color, linewidth=1.0)
        pose_arrow.set_visible(False)
        ax.add_patch(pose_arrow)
        trail_lines[spec.key] = trail_line
        pose_points[spec.key] = pose_point
        pose_arrows[spec.key] = pose_arrow

    plt.tight_layout()

    max_frames = max(len(results[spec.key].poses) for spec in method_specs)

    def init():
        artists = []
        for spec in method_specs:
            trail_lines[spec.key].set_data([], [])
            pose_points[spec.key].set_data([], [])
            pose_arrows[spec.key].set_visible(False)
            artists.extend([trail_lines[spec.key], pose_points[spec.key], pose_arrows[spec.key]])
        return artists

    def update(frame_idx: int):
        artists = []
        for spec in method_specs:
            poses = results[spec.key].poses
            idx = min(frame_idx, len(poses) - 1)
            trail_lines[spec.key].set_data(poses[: idx + 1, 0], poses[: idx + 1, 1])
            pose_points[spec.key].set_data([poses[idx, 0]], [poses[idx, 1]])

            theta = poses[idx, 2]
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            pose_arrows[spec.key].set_positions(
                (poses[idx, 0], poses[idx, 1]),
                (poses[idx, 0] + dx, poses[idx, 1] + dy),
            )
            pose_arrows[spec.key].set_visible(True)
            artists.extend([trail_lines[spec.key], pose_points[spec.key], pose_arrows[spec.key]])
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=max(1, int(round(1000.0 * DT))),
        blit=False,
        repeat=False,
    )

    fps = max(1, int(round(1.0 / DT)))
    mp4_path = output_dir / "tracking_comparison.mp4"
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=fps)
    except Exception:
        gif_path = output_dir / "tracking_comparison.gif"
        ani.save(gif_path, writer="pillow", fps=fps)

    plt.close(fig)


def write_overall_summary(output_root: Path, overall_rows: list[dict[str, str | float]]) -> None:
    summary_csv = output_root / "summary_all_paths.csv"
    headers = [
        "Path",
        "Method",
        "Constraint Violation Rate [%]",
        "Mean Position Error [m]",
        "Max Position Error [m]",
        "Mean Heading Error [deg]",
        "Travel Time [s]",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in overall_rows:
            writer.writerow(
                [
                    row["Path"],
                    row["Method"],
                    format_value(float(row["constraint_violation_rate_pct"])),
                    format_value(float(row["mean_position_error_m"])),
                    format_value(float(row["max_position_error_m"])),
                    format_value(float(row["mean_heading_error_deg"])),
                    format_value(float(row["travel_time_s"])),
                ]
            )


def get_reference_paths(args: argparse.Namespace) -> dict[str, np.ndarray]:
    one_minus_cos_num_points = args.one_minus_cos_num_points
    if one_minus_cos_num_points is None:
        one_minus_cos_num_points = 5 * args.points_per_segment + 1

    return {
        "path1_right_angle_90": right_angle_polyline_curve(
            segment_length=args.right_angle_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path2_straight_heading_step": straight_line_heading_step_curve(
            segment_length=args.heading_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path3_right_angle_90_last_heading_minus_pi": right_angle_polyline_curve_last_segment_heading_minus_pi(
            segment_length=args.right_angle_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path4_one_minus_cos": one_minus_cos_curve(
            amplitude=args.one_minus_cos_amplitude,
            length_x=args.one_minus_cos_length_x,
            num_points=one_minus_cos_num_points,
            cycles=args.one_minus_cos_cycles,
            resample_arclength=True,
        ),
    }


def get_figure_prefix(path_name: str) -> str:
    if path_name == "path3_right_angle_90_last_heading_minus_pi":
        return "exp1_"
    if path_name == "path4_one_minus_cos":
        return "exp2_"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create benchmark figures/tables from Nav2 CSV logs. "
            "Use --run multiple times: <path_name>:<method_key>=<csv_path>"
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Format: <path_name>:<method_key>=<csv_path>",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Auto-discovery mode. Directory containing Nav2 CSV logs "
            "(e.g., dwpp_nav2_YYYYMMDD_HHMMSS_NNNNNNNNN.csv). "
            "If --run is omitted, run specs are generated from this directory."
        ),
    )
    parser.add_argument(
        "--path-order",
        nargs="+",
        default=DEFAULT_AUTO_PATH_ORDER,
        help=(
            "Path names assigned to the auto-discovered runs in chronological order. "
            "Default: path3_right_angle_90_last_heading_minus_pi path4_one_minus_cos"
        ),
    )
    parser.add_argument(
        "--use-latest-runs",
        type=int,
        default=None,
        help=(
            "Number of latest complete runs to import in auto-discovery mode. "
            "Default: len(--path-order)"
        ),
    )
    parser.add_argument("--right-angle-segment-length", type=float, default=1.0)
    parser.add_argument("--heading-segment-length", type=float, default=0.5)
    parser.add_argument("--points-per-segment", type=int, default=100)
    parser.add_argument("--one-minus-cos-amplitude", type=float, default=0.75)
    parser.add_argument("--one-minus-cos-length-x", type=float, default=1.5)
    parser.add_argument("--one-minus-cos-cycles", type=float, default=1.5)
    parser.add_argument("--one-minus-cos-num-points", type=int, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_REACH_TOLERANCE_DIST_OMNI)
    parser.add_argument(
        "--goal-heading-tolerance-deg",
        type=float,
        default=float(np.rad2deg(GOAL_REACH_TOLERANCE_HEADING)),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--trajectory-frame",
        choices=["start", "map"],
        default="start",
        help=(
            "Frame used for trajectory evaluation/plotting. "
            "'start' uses robot start pose as origin (default), "
            "'map' keeps absolute map frame."
        ),
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save tracking animation (MP4/GIF). Default: off",
    )
    args = parser.parse_args()

    if len(args.run) == 0:
        if args.input_dir is None:
            raise ValueError("At least one --run is required, or specify --input-dir for auto mode.")
        args.run = discover_run_specs_from_directory(
            input_dir=args.input_dir,
            path_order=list(args.path_order),
            use_latest_runs=args.use_latest_runs,
        )

    method_keys = {m.key for m in METHOD_SPECS}
    path_to_method_csv: dict[str, dict[str, Path]] = {}
    for spec in args.run:
        path_name, method_key, csv_path = parse_run_spec(spec)
        if method_key not in method_keys:
            raise ValueError(f"Unknown method key '{method_key}'. Valid: {sorted(method_keys)}")
        path_to_method_csv.setdefault(path_name, {})[method_key] = csv_path

    ref_paths = get_reference_paths(args)
    goal_tolerance_heading = float(np.deg2rad(args.goal_heading_tolerance_deg))

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, str | float]] = []

    for path_name, method_to_csv in path_to_method_csv.items():
        if path_name not in ref_paths:
            raise ValueError(f"Unknown path name '{path_name}'. Valid: {sorted(ref_paths.keys())}")

        missing_keys = [m.key for m in METHOD_SPECS if m.key not in method_to_csv]
        if missing_keys:
            raise ValueError(
                f"Path '{path_name}' is missing CSVs for methods: {missing_keys}. "
                "Provide all methods to generate benchmark figures."
            )

        path = ref_paths[path_name]
        path_dir = output_root / path_name
        path_dir.mkdir(parents=True, exist_ok=True)
        figure_prefix = get_figure_prefix(path_name)

        np.save(path_dir / "path.npy", path)

        results: dict[str, SimulationResult] = {}
        rows: list[dict[str, str | float]] = []

        for method_spec in METHOD_SPECS:
            csv_path = method_to_csv[method_spec.key]
            result = load_nav2_csv_result(csv_path, trajectory_frame=args.trajectory_frame)
            results[method_spec.key] = result

            np.save(path_dir / f"{method_spec.key}_poses.npy", result.poses)
            np.save(path_dir / f"{method_spec.key}_velocities.npy", result.velocities_raw)
            np.save(path_dir / f"{method_spec.key}_ref_velocities.npy", result.ref_velocities_raw)
            np.save(path_dir / f"{method_spec.key}_break_flags.npy", result.break_flags)
            np.save(path_dir / f"{method_spec.key}_times.npy", result.times)

            metrics = calc_metrics(
                path,
                result,
                goal_tolerance_dist=args.goal_tolerance,
                goal_tolerance_heading=goal_tolerance_heading,
            )
            row = {"Method": method_spec.label}
            row.update(metrics)
            rows.append(row)

        write_metrics_tables(path_dir, rows, figure_prefix=figure_prefix)
        save_tracking_plots_by_method(
            path=path,
            method_specs=METHOD_SPECS,
            results=results,
            output_dir=path_dir,
            file_prefix=figure_prefix,
        )
        save_tracking_plot_overlaid(
            path=path,
            method_specs=METHOD_SPECS,
            results=results,
            output_path=path_dir / f"{figure_prefix}tracking_poses.png",
        )
        save_velocity_profiles_by_method(
            method_specs=METHOD_SPECS,
            results=results,
            output_dir=path_dir,
            file_prefix=figure_prefix,
        )
        if args.save_animation:
            save_tracking_animation(
                path=path,
                method_specs=METHOD_SPECS,
                results=results,
                output_dir=path_dir,
            )

        for row in rows:
            row_with_path = {"Path": path_name}
            row_with_path.update(row)
            overall_rows.append(row_with_path)

    write_overall_summary(output_root, overall_rows)
    print(f"[INFO] Benchmark-from-CSV finished. Output: {output_root}")


if __name__ == "__main__":
    main()
