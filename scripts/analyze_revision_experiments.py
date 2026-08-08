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
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    # ---- figures
    rep_idx = 3  # ノートブック踏襲: 4番目の試行を代表に(足りなければ末尾)
    for path_name in PATH_LIST:
        # path comparison overlay
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ref = al.reference_path[path_name]
        ax.plot(ref[:, 1], -ref[:, 0], "k--", label="Reference Path", linewidth=1, alpha=0.7)
        for controller in controllers:
            trials = all_data[path_name].get(controller, [])
            if not trials:
                continue
            tr = trials[min(rep_idx, len(trials) - 1)]
            ax.plot(tr["y"], -tr["x"], color=al.color_dict[controller], label=controller, linewidth=1)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"exp1_{path_name}_path_comparison.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # velocity profiles (MPPI のみ新規。凍結分は既存図があるため省略)
        for controller in [c for c in controllers if c == "MPPI"]:
            trials = all_data[path_name].get(controller, [])
            if not trials:
                continue
            tr = trials[min(rep_idx, len(trials) - 1)]
            df_t = tr["df"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ax1.plot(tr["t"], df_t["v_cmd"], "-", color="red", label="Reference", linewidth=1, alpha=0.8)
            ax1.plot(tr["t"], df_t["v_real"], "-", color="blue", label="Actual", linewidth=1)
            ax1.axhline(y=V_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Linear Velocity [m/s]")
            ax1.grid(True, alpha=0.3)
            ax2.plot(tr["t"], df_t["w_cmd"], "-", color="red", linewidth=1, alpha=0.8)
            ax2.plot(tr["t"], df_t["w_real"], "-", color="blue", linewidth=1)
            ax2.axhline(y=W_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
            ax2.axhline(y=-W_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Angular Velocity [rad/s]")
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"exp1_{path_name}_{controller}_velocity.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
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

def run_exp2(revision_dir: Path, out_dir: Path, summary, exp2_dir: Path = None, plan_csv: Path = None):
    log("== exp2: obstacle corridor (RPP vs DWPP, proximity heuristic ON) ==", summary)
    base = exp2_dir if exp2_dir is not None else revision_dir / "exp2_obstacle"

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
        trials = collect_trials(base, EXP2_PATH_LABEL, controller, reference, summary,
                                anchor_to_start=anchor)
        all_trials[controller] = trials
        log(f"  {EXP2_PATH_LABEL}/{controller}: {len(trials)} trials", summary)
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

    pd.DataFrame(per_trial_records).to_csv(out_dir / "exp2_per_trial_records.csv", index=False)
    write_table(pd.DataFrame(agg_records).set_index("controller").T, out_dir, "exp2_corridor_table")
    log("  NOTE: collision_flag は scan_min_dist < "
        f"{EXP2_COLLISION_DIST} m の自動判定。実験ノートの手動カウントと突き合わせること", summary)

    # ---- key figure: v_cmd + dynamic window bounds + scan_min_dist + violation
    n = sum(1 for t in all_trials.values() if t)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), sharex=False, squeeze=False)
    row = 0
    for controller in EXP2_CONTROLLERS:
        trials = all_trials.get(controller, [])
        if not trials:
            continue
        tr = trials[min(3, len(trials) - 1)]
        df_t, t = tr["df"], tr["t"]
        ax = axes[row][0]
        ax.fill_between(t, df_t["dw_v_min"], df_t["dw_v_max"], color="gray", alpha=0.3,
                        label="Dynamic window (feasible $v$)")
        ax.plot(t, df_t["v_cmd"], color=al.color_dict[controller], linewidth=1.2,
                label=f"{controller} $v_\\mathrm{{cmd}}$")
        viol = df_t["velocity_violation"].to_numpy().astype(bool)
        if viol.any():
            ax.scatter(t[viol], df_t["v_cmd"].to_numpy()[viol], s=8, color="black",
                       zorder=5, label="Constraint violation")
        if "scan_min_dist" in df_t.columns:
            ax2 = ax.twinx()
            ax2.plot(t, df_t["scan_min_dist"], color="teal", linewidth=0.8, alpha=0.7)
            ax2.set_ylabel("Min. scan distance [m]", color="teal")
            ax2.tick_params(axis="y", labelcolor="teal")
        ax.axhline(y=V_MAX, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel("Linear velocity [m/s]")
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{controller} (proximity heuristic ON)")
        row += 1
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"exp2_corridor_vcmd_window.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  exp2 figures written to {out_dir}", summary)


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
    parser.add_argument("--only", choices=["exp1", "exp2", "timing"], action="append",
                        help="指定セクションのみ実行(複数指定可)")
    parser.add_argument("--exp1-subdir", default="exp1_mppi_viz_off",
                        help="revision-dir 配下の MPPI 追従データのサブディレクトリ")
    parser.add_argument("--timing-subdirs", nargs="+",
                        default=["exp1_mppi_viz_off", "exp1_mppi_pp_time"],
                        help="計時集計に含めるサブディレクトリ (viz_on や old を除外するため明示)")
    parser.add_argument("--exp2-dir", type=Path, default=None,
                        help="実②データディレクトリ (default: <revision-dir>/exp2_obstacle)")
    parser.add_argument("--exp2-plan", type=Path,
                        default=REPO_ROOT.parent / "ytlab2_whill" / "ytlab2_whill_modules"
                        / "worlds" / "corridor" / "map" / "fixed_plan.csv",
                        help="実②の凍結参照経路 CSV (map 座標系, columns x,y[,yaw])。"
                             "見つからない場合は直線コリドー参照にフォールバック")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.revision_dir / "paper_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = args.only or ["exp1", "timing", "exp2"]

    summary = []
    log(f"frozen_dir   : {args.frozen_dir}", summary)
    log(f"revision_dir : {args.revision_dir}", summary)
    log(f"out_dir      : {out_dir}", summary)

    if "exp1" in sections:
        run_exp1(args.frozen_dir, args.revision_dir, out_dir, summary, exp1_subdir=args.exp1_subdir)
    if "timing" in sections:
        run_timing(args.revision_dir, out_dir, summary, timing_subdirs=args.timing_subdirs)
    if "exp2" in sections:
        run_exp2(args.revision_dir, out_dir, summary,
                 exp2_dir=args.exp2_dir, plan_csv=args.exp2_plan)

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nDone. Outputs in {out_dir} (see summary.txt)")


if __name__ == "__main__":
    main()
