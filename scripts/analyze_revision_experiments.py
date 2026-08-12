#!/usr/bin/env python3
"""
DWPP RAS revision 実験の一括解析スクリプト。

これ1本を叩くだけで論文用の表(CSV + LaTeX断片)と図(PDF/PNG)が out_dir に出力される:

  python3 scripts/analyze_revision_experiments.py

セクション:
  exp1   : 実①  追従指標の統合表 (凍結データ PP/APP/RPP/DWPP + 新規 MPPI) と経路比較図・速度プロファイル図
  timing : 計算時間表 (revision データの timing/*_timing.csv を手法別に集計、論文掲載用)
  exp2   : 実②  障害物環境 (Corridor) の RPP vs DWPP 比較表とキー図 (v_cmd と dynamic window 上下限)

データが無いセクションは警告してスキップする(実験前でも凍結データ部分だけ動く)。
集計は既存ノートブック dwpp_real_experiment_analysis.ipynb と同一のセマンティクス
(map_base_* を試行開始点で再アンカー、cdist 最近傍距離、mean/std は ddof=0)。
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ipynb"))
import analysis_lib as al  # noqa: E402

CONTROL_PERIOD_MS = 1000.0 / 30.0
FROZEN_CONTROLLERS = ["PP", "APP", "RPP", "DWPP"]
PATH_LIST = ["PathA", "PathB", "PathC"]
EXP2_CONTROLLERS = ["RPP", "DWPP"]
EXP2_PATH_LABEL = "Corridor"
EXP2_COLLISION_DIST = 0.25  # scan_min_dist がこれ未満なら衝突疑いフラグ
V_MAX = 0.50
W_MAX = 1.0

if "MPPI" not in al.color_dict:
    al.color_dict["MPPI"] = "purple"


def log(msg, summary_lines=None):
    print(msg)
    if summary_lines is not None:
        summary_lines.append(msg)


def corridor_reference_path(length=9.0, n_points=900):
    xs = np.linspace(0.0, length, n_points)
    return np.c_[xs, np.zeros_like(xs)]


def trial_metrics(csv_path: Path, reference: np.ndarray, anchor_to_start: bool = True) -> dict:
    """ノートブック cell 4 と同一の per-trial 指標 + 実②向け追加指標"""
    df = pd.read_csv(csv_path)

    # controller_server はゴール到達時にゼロ指令を発行し、GUIレコーダはそれも
    # 記録する(プラグイン内蔵CSVには存在しない行)。末尾の完全ゼロ指令行を
    # 除外して凍結データと意味論を揃える(急停止が違反として混入するのを防ぐ。
    # 検証済み: これで pp_time の DWPP 違反率は全経路 0.000% になり、凍結CSV
    # 60本には末尾ゼロ行が無いため凍結テーブルの再現には影響しない)。
    nonzero = df.index[(df["v_cmd"] != 0.0) | (df["w_cmd"] != 0.0)]
    if len(nonzero) and nonzero[-1] + 1 < len(df):
        df = df.loc[:nonzero[-1]]

    t = df["sec"].to_numpy() + df["nsec"].to_numpy() * 1e-9
    t = t - t[0]

    if anchor_to_start:
        # ロボット原点アンカーの経路 (実①の折れ線): 試行開始姿勢基準に変換
        x, y, _ = al.transform_pose_to_path_origin(
            df["map_base_x"].to_numpy(), df["map_base_y"].to_numpy(), df["map_base_yaw"].to_numpy(),
            df["map_base_x"].to_numpy()[0], df["map_base_y"].to_numpy()[0], df["map_base_yaw"].to_numpy()[0],
        )
    else:
        # map 座標系の固定経路 (実②の fixed_plan): map 姿勢をそのまま比較
        x = df["map_base_x"].to_numpy()
        y = df["map_base_y"].to_numpy()
    robot_path = np.vstack((x, y)).T
    tracking_errors = al.calc_tracking_error(robot_path, reference)

    metrics = {
        "csv": csv_path,
        "df": df,
        "t": t,
        "x": x,
        "y": y,
        "violation_rate": al.calc_violation_rate(df["velocity_violation"].to_numpy()),
        "tracking_errors": tracking_errors,
        "travel_time": al.measure_travel_time(t),
        "distance": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
    }
    if "scan_min_dist" in df.columns:
        scan = df["scan_min_dist"].to_numpy()
        scan = scan[np.isfinite(scan)]
        metrics["scan_min_dist_mean"] = float(scan.mean()) if scan.size else float("nan")
        metrics["scan_min_dist_min"] = float(scan.min()) if scan.size else float("nan")
    return metrics


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def write_table(df: pd.DataFrame, out_dir: Path, name: str, float_fmt="%.4f"):
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, float_format=float_fmt)
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(df.to_latex(float_format=lambda v: float_fmt % v))
    print(f"  wrote {csv_path.name} / {tex_path.name}")


def collect_trials(base_dir: Path, path_name: str, controller: str, reference, summary,
                   anchor_to_start: bool = True):
    data_dir = base_dir / path_name / controller
    trials = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        try:
            trials.append(trial_metrics(csv_path, reference, anchor_to_start=anchor_to_start))
        except Exception as exc:
            log(f"  WARN: failed to read {csv_path}: {exc}", summary)
    return trials


# ---------------------------------------------------- paper-style figure helpers

def plot_path_comparison_paper(path_name, all_data, controllers, out_dir):
    """論文 Fig 7-9(a) と同一スタイル (notebook cell 11 準拠) の経路比較図。
    全コントローラ x 全試行を重ね書きし、参照経路を最後(最前面)に描く。
    凡例は別ファイル path_comparison_label.png (make_path_comparison_legend)。"""
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    for controller in controllers:
        for tr in all_data[path_name].get(controller, []):
            ax.plot(tr["y"], tr["x"], color=al.color_dict[controller], linewidth=0.5)
    ref = al.reference_path[path_name]
    ax.plot(ref[:, 1], ref[:, 0], "k--", linewidth=1, alpha=0.7)
    ax.set_xlabel("$y$ [m]")
    ax.set_ylabel("$x$ [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{path_name}_path_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_profile_paper(tr, path_name, controller, out_dir):
    """論文 Fig 7-9(b)-(e) と同一スタイル (notebook cell 13 準拠) の速度プロファイル。
    出荷済みの凍結図は cell 20 の font.size=20 が有効な状態で生成されているため、
    ここでも 20 に合わせる (page size ≈ 472x184 pt, ω目盛 = -1/0/1)。"""
    df_t = tr["df"]
    with plt.rc_context({"font.size": 20}):
        _plot_velocity_profile_paper_inner(tr, df_t, path_name, controller, out_dir)


def _plot_velocity_profile_paper_inner(tr, df_t, path_name, controller, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.plot(tr["t"], df_t["v_cmd"], "-", color="red", linewidth=1, alpha=0.8)
    ax1.plot(tr["t"], df_t["v_real"], "-", color="blue", linewidth=1)
    ax1.axhline(y=V_MAX, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("$t$ [s]")
    ax1.set_ylabel("$v$ [m/s]")
    ax1.set_yticks([0, 0.25, 0.50])
    ax1.grid(True, alpha=0.3)
    ax2.plot(tr["t"], df_t["w_cmd"], "-", color="red", linewidth=1, alpha=0.8)
    ax2.plot(tr["t"], df_t["w_real"], "-", color="blue", linewidth=1)
    ax2.axhline(y=W_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax2.axhline(y=-W_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax2.set_xlabel("$t$ [s]")
    ax2.set_ylabel("$\\omega$ [rad/s]")
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{path_name}_{controller}_velocity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_path_comparison_legend(controllers, out_dir):
    """Fig 7-9(a) の上に置く共有凡例 (path_comparison_label.png) を再生成する。"""
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=3, label="Path")]
    handles += [Line2D([], [], color=al.color_dict[c], linewidth=3, label=c) for c in controllers]
    fig = plt.figure(figsize=(9, 0.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=True,
              edgecolor="black", handlelength=1.6, columnspacing=1.3, handletextpad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"path_comparison_label.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def load_occupancy_map(map_yaml: Path):
    """ROS の map.yaml + P5 PGM を numpy で読み、(img, extent) を返す。
    extent は map 座標系での (xmin, xmax, ymin, ymax)。PGM の行0が地図上端
    なので imshow は origin='upper' で描くこと。"""
    text = map_yaml.read_text()
    res = float(re.search(r"^resolution:\s*([\d.eE+-]+)", text, re.M).group(1))
    ox, oy = [float(v) for v in
              re.search(r"^origin:\s*\[([^\]]+)\]", text, re.M).group(1).split(",")[:2]]
    image = re.search(r"^image:\s*(\S+)", text, re.M).group(1)
    data = (map_yaml.parent / image).read_bytes()
    m = re.match(rb"P5\s+(?:#[^\n]*\s+)*(\d+)\s+(\d+)\s+(\d+)\s", data)
    w, h = int(m.group(1)), int(m.group(2))
    img = np.frombuffer(data, dtype=np.uint8, count=w * h, offset=m.end()).reshape(h, w)
    extent = (ox, ox + w * res, oy, oy + h * res)
    return img, extent


def draw_corridor_map(ax, img, extent, plan_xy):
    """corridor 図共通の地図背景 + 軸設定 (経路 x 範囲 +0.8 m マージンにクロップ)。"""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper", extent=extent,
              interpolation="nearest", zorder=0)
    ax.set_xlim(plan_xy[:, 0].min() - 0.8, plan_xy[:, 0].max() + 0.8)
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_ylabel("$y$ [m]")


# ---------------------------------------------------------------- exp1

def run_exp1(frozen_dir: Path, revision_dir: Path, out_dir: Path, summary, exp1_subdir="exp1_mppi"):
    log("== exp1: tracking metrics (frozen PP-family + revision MPPI) ==", summary)
    mppi_dir = revision_dir / exp1_subdir

    all_data = defaultdict(dict)  # [path][controller] -> list of trial metrics
    controllers = list(FROZEN_CONTROLLERS)

    for path_name in PATH_LIST:
        reference = al.reference_path[path_name]
        for controller in FROZEN_CONTROLLERS:
            trials = collect_trials(frozen_dir, path_name, controller, reference, summary)
            all_data[path_name][controller] = trials
            log(f"  {path_name}/{controller}: {len(trials)} trials (frozen)", summary)

    mppi_found = any((mppi_dir / p / "MPPI").is_dir() for p in PATH_LIST)
    if mppi_found:
        controllers.append("MPPI")
        for path_name in PATH_LIST:
            trials = collect_trials(mppi_dir, path_name, "MPPI", al.reference_path[path_name], summary)
            all_data[path_name]["MPPI"] = trials
            log(f"  {path_name}/MPPI: {len(trials)} trials (revision)", summary)
    else:
        log(f"  WARN: no MPPI data under {mppi_dir} - tables will cover the frozen four only", summary)

    # ---- aggregate tables (notebook cell 6/8 semantics)
    records = []
    for path_name in PATH_LIST:
        for controller in controllers:
            trials = all_data[path_name].get(controller, [])
            if not trials:
                continue
            vv_m, vv_s = mean_std([tr["violation_rate"] for tr in trials])
            me_m, me_s = mean_std([tr["tracking_errors"].mean() for tr in trials])
            mx_m, mx_s = mean_std([tr["tracking_errors"].max() for tr in trials])
            tt_m, tt_s = mean_std([tr["travel_time"] for tr in trials])
            records.append({
                "path": path_name, "controller": controller,
                "velocity_violation_mean": vv_m, "velocity_violation_std": vv_s,
                "mean_tracking_errors_mean": me_m, "mean_tracking_errors_std": me_s,
                "max_tracking_errors_mean": mx_m, "max_tracking_errors_std": mx_s,
                "travel_time_mean": tt_m, "travel_time_std": tt_s,
                "n_trials": len(trials),
            })
    if not records:
        log("  WARN: no exp1 data at all - skipped", summary)
        return
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "exp1_per_condition_records.csv", index=False)

    col_order = [c for c in controllers]
    for value, name in [
        ("velocity_violation_mean", "velocity_violation_mean_table"),
        ("velocity_violation_std", "velocity_violation_std_table"),
        ("mean_tracking_errors_mean", "all_tracking_errors_mean_table"),
        ("mean_tracking_errors_std", "all_tracking_errors_std_table"),
        ("max_tracking_errors_mean", "max_tracking_errors_mean_table"),
        ("max_tracking_errors_std", "max_tracking_errors_std_table"),
        ("travel_time_mean", "travel_time_mean_table"),
        ("travel_time_std", "travel_time_std_table"),
    ]:
        table = df.pivot(index="path", columns="controller", values=value)
        table = table.reindex(columns=[c for c in col_order if c in table.columns])
        write_table(table, out_dir, name)

    # ---- figures (論文 Fig 7-9 スタイル: 全試行の経路比較 + MPPI 速度プロファイル)
    rep_idx = 3  # ノートブック踏襲: 4番目の試行を代表に(足りなければ末尾)
    for path_name in PATH_LIST:
        plot_path_comparison_paper(path_name, all_data, controllers, out_dir)
        for controller in [c for c in controllers if c == "MPPI"]:
            trials = all_data[path_name].get(controller, [])
            if not trials:
                continue
            tr = trials[min(rep_idx, len(trials) - 1)]
            plot_velocity_profile_paper(tr, path_name, controller, out_dir)
    make_path_comparison_legend(controllers, out_dir)
    log(f"  exp1 figures written to {out_dir}", summary)


# ---------------------------------------------------------------- timing

def run_timing(revision_dir: Path, out_dir: Path, summary, timing_subdirs=None):
    """timing_subdirs を明示することで、破棄試行 (old/) や viz_on 版など
    論文に使わないディレクトリの計時が混入するのを防ぐ"""
    log("== timing: per-controller computation time ==", summary)
    subdirs = timing_subdirs or ["exp1_mppi", "exp1_mppi_pp_time"]
    timing_files = []
    for sub in subdirs:
        d = revision_dir / sub
        if d.is_dir():
            found = sorted(p for p in d.glob("**/timing/*_timing.csv") if "old" not in p.parts)
            timing_files += found
            log(f"  {sub}: {len(found)} timing files", summary)
        else:
            log(f"  WARN: timing subdir missing: {d}", summary)
    if not timing_files:
        log(f"  WARN: no timing CSVs under {revision_dir} - skipped", summary)
        return
    frames = []
    for f in timing_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f.relative_to(revision_dir))
            frames.append(df)
        except Exception as exc:
            log(f"  WARN: failed to read {f}: {exc}", summary)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    records = []
    for controller, g in df.groupby("controller_id"):
        ok = g[g["success"].astype(bool)] if "success" in g.columns else g
        ms = ok["compute_time_ms"].to_numpy(dtype=float)
        records.append({
            "controller": controller,
            "n_trials": g["source_file"].nunique(),
            "n_cycles": len(ms),
            "mean_ms": ms.mean(),
            "std_ms": ms.std(ddof=0),
            "max_ms": ms.max(),
            "p99_ms": float(np.percentile(ms, 99)),
            "mean_ratio_to_period": ms.mean() / CONTROL_PERIOD_MS,
        })
    table = pd.DataFrame(records).set_index("controller")
    # 論文の手法順に並べる
    order = [c for c in FROZEN_CONTROLLERS + ["MPPI"] if c in table.index]
    table = table.reindex(order + [c for c in table.index if c not in order])
    write_table(table, out_dir, "computation_time_table")
    log(f"  pooled {len(df)} cycles from {len(timing_files)} trials "
        f"(control period {CONTROL_PERIOD_MS:.2f} ms)", summary)


# ---------------------------------------------------------------- exp2

def run_exp2(revision_dir: Path, out_dir: Path, summary, exp2_dir: Path = None, plan_csv: Path = None,
             map_yaml: Path = None, label: str = EXP2_PATH_LABEL):
    log("== exp2: obstacle corridor (RPP vs DWPP, proximity heuristic ON) ==", summary)
    base = exp2_dir if exp2_dir is not None else REPO_ROOT / "data" / "exp2_obstacle"

    if plan_csv is not None and Path(plan_csv).is_file():
        # 凍結済み map 座標系の参照経路 (実験で全試行に共通配信した fixed plan)
        plan = pd.read_csv(plan_csv)
        reference = np.c_[plan["x"].to_numpy(), plan["y"].to_numpy()]
        anchor = False
        log(f"  reference: fixed plan {plan_csv} ({len(reference)} pts, map frame)", summary)
    else:
        reference = corridor_reference_path()
        anchor = True
        log("  reference: straight 9 m corridor line (start-anchored)", summary)

    all_trials = {}
    for controller in EXP2_CONTROLLERS:
        trials = collect_trials(base, label, controller, reference, summary,
                                anchor_to_start=anchor)
        all_trials[controller] = trials
        log(f"  {label}/{controller}: {len(trials)} trials", summary)
    if not any(all_trials.values()):
        log(f"  WARN: no exp2 data under {base} - skipped", summary)
        return

    # ---- per-trial + aggregate table (RPP論文 Table 2 準拠 + 制約違反率)
    per_trial_records = []
    agg_records = []
    for controller, trials in all_trials.items():
        if not trials:
            continue
        for i, tr in enumerate(trials):
            per_trial_records.append({
                "controller": controller, "trial": i,
                "time_s": tr["travel_time"],
                "distance_m": tr["distance"],
                "avg_speed_mps": tr["distance"] / tr["travel_time"] if tr["travel_time"] > 0 else float("nan"),
                "avg_dist_to_path_m": float(tr["tracking_errors"].mean()),
                "avg_dist_to_obstacle_m": tr.get("scan_min_dist_mean", float("nan")),
                "min_dist_to_obstacle_m": tr.get("scan_min_dist_min", float("nan")),
                "collision_flag": int(tr.get("scan_min_dist_min", float("inf")) < EXP2_COLLISION_DIST),
                "violation_rate_pct": tr["violation_rate"],
            })
        rec = {"controller": controller, "n_trials": len(trials)}
        for key, values in [
            ("time_s", [tr["travel_time"] for tr in trials]),
            ("distance_m", [tr["distance"] for tr in trials]),
            ("avg_speed_mps", [tr["distance"] / tr["travel_time"] for tr in trials]),
            ("avg_dist_to_path_m", [float(tr["tracking_errors"].mean()) for tr in trials]),
            ("avg_dist_to_obstacle_m", [tr.get("scan_min_dist_mean", float("nan")) for tr in trials]),
            ("min_dist_to_obstacle_m", [tr.get("scan_min_dist_min", float("nan")) for tr in trials]),
            ("violation_rate_pct", [tr["violation_rate"] for tr in trials]),
        ]:
            m, s = mean_std(values)
            rec[f"{key}_mean"] = m
            rec[f"{key}_std"] = s
        rec["collisions_total"] = sum(
            1 for tr in trials if tr.get("scan_min_dist_min", float("inf")) < EXP2_COLLISION_DIST
        )
        agg_records.append(rec)

    stem = f"exp2_{label.lower()}"
    pd.DataFrame(per_trial_records).to_csv(out_dir / f"{stem}_per_trial_records.csv", index=False)
    write_table(pd.DataFrame(agg_records).set_index("controller").T, out_dir, f"{stem}_table")
    log("  NOTE: collision_flag は scan_min_dist < "
        f"{EXP2_COLLISION_DIST} m の自動判定。実験ノートの手動カウントと突き合わせること", summary)

    # ---- figures
    rep_trials = {c: trials[min(3, len(trials) - 1)]
                  for c, trials in all_trials.items() if trials}

    # (1)(2) 地図ベースの図は map.yaml と fixed_plan がある場合のみ
    if map_yaml is not None and Path(map_yaml).is_file() and not anchor:
        img, extent = load_occupancy_map(Path(map_yaml))
        plot_corridor_map_plan(reference, img, extent, out_dir, stem)
        plot_corridor_speed_colored(rep_trials, reference, img, extent, out_dir, stem)
    else:
        log(f"  WARN: map yaml missing or no fixed plan ({map_yaml}) - "
            "map-based corridor figures skipped", summary)

    # (3) 速度指令・曲率指令・障害物距離の 2x3 パネル
    plot_corridor_vcmd_panels(rep_trials, out_dir, stem)
    log(f"  exp2 figures written to {out_dir}", summary)


def plot_corridor_map_plan(plan_xy, img, extent, out_dir, stem):
    """実②の設定図: 占有格子地図 + 大域経路 (fixed plan) + start/goal。"""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    draw_corridor_map(ax, img, extent, plan_xy)
    ax.plot(plan_xy[:, 0], plan_xy[:, 1], "k--", linewidth=1.5, label="Reference path", zorder=2)
    ax.plot(plan_xy[0, 0], plan_xy[0, 1], "o", color="limegreen", mec="black", ms=8,
            zorder=4, label="Start")
    ax.plot(plan_xy[-1, 0], plan_xy[-1, 1], "*", color="red", mec="black", ms=12,
            zorder=4, label="Goal")
    ax.set_xlabel("$x$ [m]")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_map_plan.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_corridor_speed_colored(rep_trials, plan_xy, img, extent, out_dir, stem):
    """実②の追従軌跡を実現速度 v_real で色付けした図 (RPP 上段 / DWPP 下段)。"""
    controllers = [c for c in EXP2_CONTROLLERS if c in rep_trials]
    fig, axes = plt.subplots(len(controllers), 1, figsize=(8, 2.6 * len(controllers)),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    norm = plt.Normalize(0.0, V_MAX)
    lc = None
    for row, controller in enumerate(controllers):
        tr = rep_trials[controller]
        ax = axes[row][0]
        draw_corridor_map(ax, img, extent, plan_xy)
        ax.plot(plan_xy[:, 0], plan_xy[:, 1], "k--", linewidth=1, alpha=0.7, zorder=2)
        v = tr["df"]["v_real"].to_numpy(dtype=float)
        pts = np.column_stack([tr["x"], tr["y"]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_v = 0.5 * (v[:-1] + v[1:])
        ok = np.isfinite(seg_v) & np.isfinite(segs).all(axis=(1, 2))
        lc = LineCollection(segs[ok], cmap="turbo", norm=norm, linewidth=2.5,
                            capstyle="round", zorder=3)
        lc.set_array(seg_v[ok])
        ax.add_collection(lc)
        ax.set_title(controller, loc="left", fontsize=12)
    axes[-1][0].set_xlabel("$x$ [m]")
    fig.colorbar(lc, ax=[axes[r][0] for r in range(len(controllers))],
                 orientation="vertical", fraction=0.03, pad=0.02,
                 label="Linear velocity [m/s]")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_speed_colored.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_corridor_vcmd_panels(rep_trials, out_dir, stem):
    """実②のキー図: 行=コントローラ (RPP/DWPP)、列=(1) 線速度指令+実現+動的窓
    (2) 曲率指令 kappa = w_cmd/v_cmd (3) 最小障害物距離。横軸は時間。"""
    controllers = [c for c in EXP2_CONTROLLERS if c in rep_trials]
    fig, axes = plt.subplots(len(controllers), 3, figsize=(10, 2.4 * len(controllers)),
                             sharex="row", squeeze=False)
    for row, controller in enumerate(controllers):
        tr = rep_trials[controller]
        df_t, t = tr["df"], tr["t"]
        color = al.color_dict[controller]

        ax = axes[row][0]
        ax.fill_between(t, df_t["dw_v_min"], df_t["dw_v_max"], color="gray", alpha=0.3,
                        label="Dynamic window")
        ax.plot(t, df_t["v_cmd"], color=color, linewidth=1.2, label="$v_\\mathrm{cmd}$")
        ax.plot(t, df_t["v_real"], color="blue", linewidth=1, alpha=0.9,
                label="$v_\\mathrm{real}$")
        viol = df_t["velocity_violation"].to_numpy().astype(bool)
        if viol.any():
            ax.scatter(t[viol], df_t["v_cmd"].to_numpy()[viol], s=8, color="black",
                       zorder=5, label="Violation")
        ax.axhline(y=V_MAX, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel(f"{controller}\nLinear velocity [m/s]")
        ax.legend(loc="lower center", fontsize=7, ncol=2, framealpha=0.9)

        ax = axes[row][1]
        v_cmd = df_t["v_cmd"].to_numpy(dtype=float)
        w_cmd = df_t["w_cmd"].to_numpy(dtype=float)
        kappa = np.where(np.abs(v_cmd) >= 0.05,
                         w_cmd / np.where(v_cmd == 0.0, np.nan, v_cmd), np.nan)
        ax.plot(t, kappa, color=color, linewidth=1.2)
        # 曲率ヒューリスティック閾値 |kappa| = 1/R_min (R_min = 0.9 m)
        kappa_th = 1.0 / 0.9
        ax.axhline(y=kappa_th, color="gray", linestyle="--", linewidth=1,
                   label="$\\pm 1/R_\\mathrm{min}$")
        ax.axhline(y=-kappa_th, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Commanded curvature [1/m]")
        ax.set_ylim(-1.3, 1.3)
        if row == 0:
            ax.legend(loc="lower right", fontsize=8)

        ax = axes[row][2]
        ax.plot(t, df_t["scan_min_dist"], color="teal", linewidth=1.2)
        ax.axhline(y=0.6, color="gray", linestyle="--", linewidth=1,
                   label="$d_\\mathrm{prox}$")
        ax.set_ylabel("Min. obstacle distance [m]")
        ax.set_ylim(bottom=0)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)

        for col in range(3):
            axes[row][col].grid(True, alpha=0.3)
            if row == len(controllers) - 1:
                axes[row][col].set_xlabel("Time [s]")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_vcmd_window.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-dir", type=Path, default=REPO_ROOT / "data" / "real_robot_experiment",
                        help="凍結済み論文データ (PP/APP/RPP/DWPP)")
    parser.add_argument("--revision-dir", type=Path,
                        default=REPO_ROOT / "data" / "real_robot_experiment_revision",
                        help="revision 実験データルート (exp1_mppi / exp2_obstacle)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="出力先 (default: <revision-dir>/paper_outputs)")
    parser.add_argument("--only", choices=["exp1", "exp2", "timing", "legends"], action="append",
                        help="指定セクションのみ実行(複数指定可)。legends は共有凡例のみ再生成")
    parser.add_argument("--exp1-subdir", default="exp1_mppi_viz_off",
                        help="revision-dir 配下の MPPI 追従データのサブディレクトリ")
    parser.add_argument("--timing-subdirs", nargs="+",
                        default=["exp1_mppi_viz_off", "exp1_mppi_pp_time"],
                        help="計時集計に含めるサブディレクトリ (viz_on や old を除外するため明示)")
    parser.add_argument("--exp2-dir", type=Path, default=REPO_ROOT / "data" / "exp2_obstacle",
                        help="実②データディレクトリ")
    parser.add_argument("--exp2-label", default="Corridor",
                        help="実②のコースラベル (サブディレクトリ名, e.g. Corridor / Corridor_R1.5)")
    parser.add_argument("--exp2-plan", type=Path,
                        default=REPO_ROOT.parent / "ytlab2_whill" / "ytlab2_whill_modules"
                        / "worlds" / "corridor" / "map" / "fixed_plan.csv",
                        help="実②の凍結参照経路 CSV (map 座標系, columns x,y[,yaw])。"
                             "見つからない場合は直線コリドー参照にフォールバック")
    parser.add_argument("--exp2-map", type=Path, default=None,
                        help="実②の占有格子地図 map.yaml (default: --exp2-plan と同ディレクトリの map.yaml)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.revision_dir / "paper_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = args.only or ["exp1", "timing", "exp2"]

    summary = []
    log(f"frozen_dir   : {args.frozen_dir}", summary)
    log(f"revision_dir : {args.revision_dir}", summary)
    log(f"out_dir      : {out_dir}", summary)

    exp2_map = args.exp2_map or args.exp2_plan.parent / "map.yaml"

    if "exp1" in sections:
        run_exp1(args.frozen_dir, args.revision_dir, out_dir, summary, exp1_subdir=args.exp1_subdir)
    if "timing" in sections:
        run_timing(args.revision_dir, out_dir, summary, timing_subdirs=args.timing_subdirs)
    if "exp2" in sections:
        run_exp2(args.revision_dir, out_dir, summary,
                 exp2_dir=args.exp2_dir, plan_csv=args.exp2_plan, map_yaml=exp2_map,
                 label=args.exp2_label)
    if "legends" in sections and "exp1" not in sections:
        make_path_comparison_legend(FROZEN_CONTROLLERS + ["MPPI"], out_dir)
        log("== legends: path_comparison_label regenerated ==", summary)

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nDone. Outputs in {out_dir} (see summary.txt)")


if __name__ == "__main__":
    main()
