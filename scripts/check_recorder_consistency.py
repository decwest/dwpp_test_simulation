#!/usr/bin/env python3
"""
GUIレコーダCSV と プラグイン内蔵ロガーCSV の整合性チェック(開発検証用)。

同一走行(PP系コントローラ)について、
  - 最近傍時刻結合 (許容 33 ms)
  - v_cmd / w_cmd の一致 (許容 1e-6)
  - velocity_violation の一致率 > 99%
を検証する。合格で exit 0、不合格で exit 1。

usage:
  python3 scripts/check_recorder_consistency.py \
      --recorder-csv data/mppi_obstacle_experiment/.../PathA_DWPP_*.csv \
      --plugin-csv   <install share>/dwpp_test_simulation/data/dynamic_window_pure_pursuit_log_*.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

JOIN_TOL_S = 0.033
CMD_TOL = 1e-6
VIOLATION_AGREEMENT_MIN = 0.99


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["t"] = df["sec"].astype(float) + df["nsec"].astype(float) * 1e-9
    return df.sort_values("t").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recorder-csv", required=True)
    parser.add_argument("--plugin-csv", required=True)
    args = parser.parse_args()

    rec = load(args.recorder_csv)
    plg = load(args.plugin_csv)
    print(f"recorder rows: {len(rec)}, plugin rows: {len(plg)}")

    merged = pd.merge_asof(
        rec, plg, on="t", direction="nearest",
        tolerance=JOIN_TOL_S, suffixes=("_rec", "_plg"),
    ).dropna(subset=["v_cmd_plg"])
    print(f"joined rows (<= {JOIN_TOL_S * 1e3:.0f} ms): {len(merged)} "
          f"({len(merged) / max(len(rec), 1) * 100:.1f}% of recorder rows)")
    if len(merged) < 0.9 * min(len(rec), len(plg)):
        print("FAIL: join coverage below 90% - clocks or row cadence mismatch")
        sys.exit(1)

    ok = True

    for col in ["v_cmd", "w_cmd"]:
        diff = (merged[f"{col}_rec"] - merged[f"{col}_plg"]).abs()
        max_diff = float(diff.max())
        n_bad = int((diff > CMD_TOL).sum())
        status = "OK" if n_bad == 0 else "FAIL"
        print(f"{col}: max |diff| = {max_diff:.2e}, rows > {CMD_TOL:g}: {n_bad} -> {status}")
        ok &= n_bad == 0

    agree = (
        merged["velocity_violation_rec"].astype(int)
        == merged["velocity_violation_plg"].astype(int)
    ).mean()
    status = "OK" if agree > VIOLATION_AGREEMENT_MIN else "FAIL"
    print(f"velocity_violation agreement: {agree * 100:.2f}% "
          f"(threshold {VIOLATION_AGREEMENT_MIN * 100:.0f}%) -> {status}")
    ok &= agree > VIOLATION_AGREEMENT_MIN

    # 参考情報 (合否には含めない): v_nav はロック外キャッシュ由来のずれが乗り得る,
    # dw_* はプラグイン側が regulation 適用後のため regulation 発動区間で不一致になる
    for col in ["v_nav", "w_nav", "dw_v_max", "dw_v_min"]:
        c_rec, c_plg = f"{col}_rec", f"{col}_plg"
        if c_rec in merged.columns and c_plg in merged.columns:
            diff = (merged[c_rec] - merged[c_plg]).abs()
            print(f"[info] {col}: max |diff| = {float(diff.max()):.4f}, "
                  f"mean = {float(diff.mean()):.5f}")

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
