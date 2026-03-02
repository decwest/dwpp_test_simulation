from tkinter import filedialog
import glob
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def step_curves() -> list[np.ndarray]:
    """Reference paths defined in start-pose local frame."""
    paths = []
    theta_list = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    l = 3.0

    for theta in theta_list:
        x1 = np.linspace(0, 1, 100)
        y1 = np.zeros_like(x1)

        x2 = np.linspace(1.0, 1.0 + l * math.cos(theta), 100)
        y2 = np.linspace(0.0, l * math.sin(theta), 100)

        x3 = np.linspace(1.0 + l * math.cos(theta), 4.0 + l * math.cos(theta), 100)
        y3 = np.ones_like(x3) * l * math.sin(theta)

        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])
        paths.append(np.c_[x, y])

    return paths


def calc_rmse(robot_path: np.ndarray, path: np.ndarray) -> float:
    distance_matrix = cdist(robot_path, path, metric="euclidean")
    min_distances = np.min(distance_matrix, axis=1)
    return float(np.sqrt(np.mean(min_distances ** 2)))


def calc_violation_rate(violation_flags: np.ndarray) -> float:
    total_count = len(violation_flags)
    if total_count == 0:
        return 0.0
    violation_count = np.sum(violation_flags)
    return float(violation_count / total_count)


def to_start_frame_xy(x: np.ndarray, y: np.ndarray, yaw: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert trajectory to start-pose frame:
      - origin at first sample (x0, y0)
      - x-axis aligned with first yaw (if yaw is available)
    """
    x0 = float(x[0])
    y0 = float(y[0])
    dx = x - x0
    dy = y - y0

    if yaw is None:
        return dx, dy

    yaw0 = float(yaw[0])
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return x_local, y_local


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> np.ndarray | None:
    for c in candidates:
        if c in df.columns:
            return df[c].values
    return None


def infer_path_name(filepath: str) -> str:
    parent = os.path.basename(os.path.dirname(filepath))
    if parent in {"PathA", "PathB", "PathC"}:
        return parent

    fname = os.path.basename(filepath)
    if ("Path A" in fname) or ("PathA" in fname):
        return "PathA"
    if ("Path B" in fname) or ("PathB" in fname):
        return "PathB"
    if ("Path C" in fname) or ("PathC" in fname):
        return "PathC"
    return parent


def main():
    filedir = filedialog.askdirectory(
        initialdir="/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb"
    )
    if not filedir:
        print("No directory selected.")
        return

    data_paths = sorted(glob.glob(os.path.join(filedir, "*.csv")))
    if not data_paths:
        print(f"No CSV files found in: {filedir}")
        return

    pathA, pathB, pathC = step_curves()
    path_dict = {"PathA": pathA, "PathB": pathB, "PathC": pathC}

    for filepath in data_paths:
        print("Processing file:", filepath)
        df = pd.read_csv(filepath)

        if ("x" not in df.columns) or ("y" not in df.columns):
            print("  Skip: x/y columns are missing.")
            continue

        path_name = infer_path_name(filepath)
        method_name = os.path.basename(filepath).split("_")[0]

        t = first_existing_column(df, ["t", "sec"])
        if t is None:
            t = np.arange(len(df), dtype=float)
        t = t - t[0]

        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        yaw = df["yaw"].values.astype(float) if "yaw" in df.columns else None
        x_local, y_local = to_start_frame_xy(x, y, yaw)
        robot_path = np.c_[x_local, y_local]

        velocity_violation = first_existing_column(df, ["velocity_violation"])
        if velocity_violation is None:
            velocity_violation = np.zeros(len(df), dtype=bool)

        if path_name in path_dict:
            rmse = calc_rmse(robot_path, path_dict[path_name])
            print(f"  RMSE ({path_name}, start-frame): {rmse:.4f} m")
        else:
            rmse = float("nan")
            print(f"  RMSE skipped: unknown path_name='{path_name}'")

        violation_rate = calc_violation_rate(velocity_violation.astype(float))
        print(f"  Violation Rate: {violation_rate:.4f}")

        txt_filename = os.path.join(os.path.dirname(filepath), "result.txt")
        with open(txt_filename, "a", encoding="utf-8") as f:
            f.write(f"File: {os.path.basename(filepath)}\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Path: {path_name}\n")
            f.write(f"RMSE(start-frame): {rmse:.4f} m\n")
            f.write(f"Violation Rate: {violation_rate:.4f}\n\n")

        v = first_existing_column(df, ["v", "v_real"])
        w = first_existing_column(df, ["w", "w_real"])
        cmd_v = first_existing_column(df, ["v_cmd", "v_nav"])
        cmd_w = first_existing_column(df, ["w_cmd", "w_nav"])

        if (v is not None) and (cmd_v is not None):
            plt.figure(figsize=(10, 6))
            plt.plot(t, v, label="actual", color="blue")
            plt.plot(t, cmd_v, label="reference", color="red")
            plt.xlabel("Time [s]")
            plt.ylabel("Linear Velocity [m/s]")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{os.path.dirname(filepath)}/{method_name}_velocity_profile.png")
            plt.close()

        if (w is not None) and (cmd_w is not None):
            plt.figure(figsize=(10, 6))
            plt.plot(t, w, label="actual", color="blue")
            plt.plot(t, cmd_w, label="reference", color="red")
            plt.xlabel("Time [s]")
            plt.ylabel("Angular Velocity [rad/s]")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{os.path.dirname(filepath)}/{method_name}_angular_velocity_profile.png")
            plt.close()

        plt.figure(figsize=(8, 8))
        if path_name in path_dict:
            ref = path_dict[path_name]
            plt.plot(ref[:, 0], ref[:, 1], "--", color="black", label=f"{path_name} ref")
        plt.plot(x_local, y_local, color="blue", label=f"{method_name} traj")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x [m] (start frame)")
        plt.ylabel("y [m] (start frame)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{os.path.dirname(filepath)}/{method_name}_trajectory_start_frame.png")
        plt.close()


if __name__ == "__main__":
    main()
