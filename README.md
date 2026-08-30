# dwpp_test_simulation

DWPP の ROS 実験スタック(シミュレーション・実機のレコーダ、実験用 params)。

- 実機レコーダ: `scripts/follow_path_test_gui_real.py`(コントローラ非依存 35列 + scan_min_dist)
- 実験用パラメータ:
  - `params/path_tracking_experiment.yaml`(経路追従比較実験)
  - `params/confined_corridor_experiment.yaml`(障害物環境での経路追従実験)

本リポジトリは実験の実施(データ取得)用。
