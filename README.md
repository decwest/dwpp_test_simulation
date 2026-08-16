# dwpp_test_simulation

DWPP の ROS 実験スタック(シミュレーション・実機のレコーダ、実験用 params、手順書)。

- 改訂実験の手順書: `docs/revision_experiment_runbook.md`
- 実機レコーダ: `scripts/follow_path_test_gui_real.py`(コントローラ非依存 35列 + scan_min_dist)

## 論文用データと解析の所在(2026-08-16 移設)

DWPP 論文(RAS revision)の凍結実験データと解析スクリプト
`analyze_revision_experiments.py` は、論文リポジトリ
[Decwest/DWPP_RAS](https://github.com/Decwest/DWPP_RAS) の `data/` と `scripts/` に移した。
論文図表の再現はそちらで完結する。本リポジトリは実験の実施(データ取得)専用。
