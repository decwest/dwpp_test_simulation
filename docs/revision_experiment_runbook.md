# DWPP RAS Revision 実験手順書(実①MPPI比較・実②障害物環境)

対象: ROBOT-D-26-00153 major revision 対応の追加実機実験。
場所: cafeteria3F(2025/12 の論文実験と同一マップ)。

## 0. 前提(準備済みの内容)

| 項目 | 内容 |
|------|------|
| navigation2 ブランチ | `feature/revision_experiments_jazzy` = `feature/test_dwpp_jazzy`(論文実験時のDWPP fork+35列ロガー)+ 計算時間発行(`/controller_server/computation_time`, cherry-pick b7d93bad) |
| ソースビルド対象 | `nav2_msgs` / `nav2_controller` / `nav2_regulated_pure_pursuit_controller`(`docker/initial_setting.sh` の `cb` 例外リストに追加済み) |
| MPPI | apt 版 `ros-jazzy-nav2-mppi-controller`(`/opt/ros/jazzy`)。計算時間計測は controller_server 側なので MPPI にも効く |
| レコーダ | `scripts/follow_path_test_gui_real.py` に**コントローラ非依存**の記録を実装。`/cmd_vel_nav` 受信駆動で1周期1行、プラグインロガーと同一の35列+`scan_min_dist`。violation / dynamic window / v_nav は C++ と同一ロジックの Python 移植で再計算 |
| 計時 CSV | `/controller_server/computation_time` を購読し `timing/` サブディレクトリに保存(**論文の計算時間表の元データ**) |
| 凍結データ | `data/backup/real_robot_experiment_paper_freeze_20260725` にバックアップ済み。新データは `data/real_robot_experiment_revision/` 配下のみ |

## 1. ビルド(実験日の朝に必ず確認)

```bash
cd ~/decwest_workspace/ecpp_ws/ytlab2_whill
git submodule status third_party/navigation2   # feature/revision_experiments_jazzy (531e41ca) を指すこと
docker compose run --rm whill_jazzy_cpu bash   # または通常の task run
# コンテナ内:
source docker/initial_setting.sh 相当(自動)
cb                                              # nav2_msgs / nav2_controller / RPP fork がビルドされる
ros2 pkg prefix nav2_regulated_pure_pursuit_controller   # ~/ros2_ws/install を指すこと(aptではなく)
ros2 interface show nav2_msgs/msg/ControllerComputation  # 表示されること
```

## 2. 実①: MPPI 比較(45/90/135° 折れ線)

### 起動

```bash
ros2 launch ytlab2_whill_modules dwpp_experiment.launch.py \
  params_path:=/home/ubuntu/ros2_ws/src/dwpp_test_simulation/params/revision_exp1_mppi_params.yaml \
  experiment_name:=real_robot_experiment_revision/exp1_mppi \
  path_set:=polyline
```

### 起動後チェック(トラップ検出)

```bash
ros2 param get /controller_server controller_plugins   # ['PP','APP','RPP','DWPP','MPPI'] であること
                                                       # (nelson デフォルトのままだと違うリストになる)
ros2 topic echo /controller_server/computation_time --once   # 走行開始後に流れること
```

Nav2 の lifecycle bringup は bond タイムアウトで間欠的に途中放棄されることがあるが、
launch に組み込みの `nav2_bringup_watchdog`(起動20秒後に動作)が未activateノードを
自動修復する。ログに `all lifecycle nodes active - watchdog done` が出れば bringup 完了。
`gave up:` が出た場合のみ launch を再起動する。
- RViz で経路プレビューが **45/90/135° 折れ線**であること(ISO 正方形なら path_set 未指定)。
- AMCL 収束確認: GUI が起動時に (0,0,0) の initialpose を自動発行するので、**2D Pose Estimate で再初期化 → スキャンが壁に一致してから** "Update Path Origin"。

### 走行メニュー

| 目的 | 内容 |
|------|------|
| MPPI 追従評価 | MPPI × {PathA, PathB, PathC} × 5 試行 |
| 計算時間表(論文掲載) | PP / APP / RPP / DWPP × PathC × 5 走行(計時が目的。追従指標の論文表は凍結データのまま。余裕があれば3経路) |
| 再現性サニティ | DWPP × 各経路1本(凍結データと同傾向であることの確認用) |

- 各試行後、`data/real_robot_experiment_revision/exp1_mppi/{Path}/{ctrl}/` に CSV と `timing/*_timing.csv` が出ていること・行数 ≈ 30×走行秒数であることを確認(レコーダが行間隔・計時欠落を警告する)。
- MPPI が低加速度制限で振動する場合: `vx_std: 0.20 / wz_std: 0.40` → `0.15 / 0.30` にフォールバック(params のコメント参照)。**採用値を実験ノートに記録**。
- バッテリー残量を試行ごとにメモ(>50% 推奨)。

## 3. 実②: 障害物環境(corridor スラローム、RPP vs DWPP)

### パラメータ方針(RPP 論文との対応)

- **速度・加速度制約・lookahead は凍結値のまま**(RPP 論文は別ロボット(Tiago)での実験のため、v_max 0.8 / a_max 0.2 / lookahead 0.25–1.2 等は採用しない)。
- **RPP のヒューリスティクス関連パラメータは論文に数値記載が一切ない**(r_min、proximity の d_prox・α、最低速度閾値とも)。本実験の値(cost_scaling_dist 0.6 / cost_scaling_gain 1.0 / inflation_cost_scaling_factor 3.0 / r_min 0.90 / min_speed 0.25)は **RPP 著者自身の Nav2 リファレンス実装デフォルト**であり、可能な範囲で最も論文に忠実な設定。レターでもそのように説明する。
- **コース幾何のみ論文踏襲**(幅 1.5 m、障害物 ~0.7 m、5 試行、指標)。
- レコーダの制約値は params_path から自動同期される(起動ログの `Recorder limits synced from ...` を確認)。

### コース設営(RPP 論文 confined corridor 踏襲)

- 幅 **1.5 m** の通路(パネル/段ボール壁)。
- 障害物 3 個(幅 ~**0.7 m**、段ボール等の軟質物)を左右交互に配置。
- 経路(通路中心線)との側方クリアランス ~**0.3 m**(cost_scaling_dist 0.6 未満 → 減速発動、robot_radius 0.22 超 → 経路上は接触なし)。
- 障害物への局所的な反応はライブスキャンで local costmap に入る。作成済みの
  `worlds/corridor/map/map.yaml` は AMCL と NavFn の大域計画に使用する。

### 起動

```bash
ros2 launch ytlab2_whill_modules dwpp_experiment.launch.py \
  map_path:=/home/ubuntu/ros2_ws/src/ytlab2_whill_modules/worlds/corridor/map/map.yaml \
  params_path:=/home/ubuntu/ros2_ws/src/dwpp_test_simulation/params/revision_exp2_obstacle_params.yaml \
  experiment_name:=real_robot_experiment_revision/exp2_obstacle \
  path_set:=corridor
```

チェック: `ros2 param get /controller_server controller_plugins` → `['RPP','DWPP']`、
`ros2 param get /controller_server RPP.use_cost_regulated_linear_velocity_scaling` → `true`。

`Corridor` の初回実行時だけ、NavFn が map 座標の `(0,0,0)` から
ゴール姿勢 `(10.0,-0.2,0 deg)` までを計画し、
`worlds/corridor/map/fixed_plan.csv` に保存する。以後は再計画せず、RPP と
DWPP の全試行でこの CSV を読み込んだ同一経路を使用する。地図を変更して
再計画する場合は、既存の `fixed_plan.csv` を退避してから起動する。

### ⚠️ 安全(重要)

- この設定では **collision_monitor の介入を全て無効化**している(比較の交絡回避のため)。
- **監督者が WHILL のジョイスティックに手を添えて随伴**(ジョイスティック操作は制御より優先)+ 無線E-stop。
- 障害物は軟質物のみ。通路内に人が入らないこと。

### 走行メニュー

- RPP × Corridor × 5 試行、DWPP × Corridor × 5 試行。
- 衝突(接触)があれば**手動でカウントして実験ノートに記録**(解析の collision_flag は scan_min_dist < 0.25 m の自動判定であり、通路壁の誤検知があり得るため手動記録が正)。

## 4. 解析(1コマンド)

```bash
cd ~/decwest_workspace/ecpp_ws/dwpp_test_simulation   # ホスト側でOK (Python 3.8+)
python3 scripts/analyze_revision_experiments.py
```

出力: `data/real_robot_experiment_revision/paper_outputs/` に
- 実①統合表(凍結 PP系 + MPPI の 5手法×3経路、既存ノートブックと同一集計。凍結分のみで実行しても既存テーブルと一致することを検証済み)
- `computation_time_table.{csv,tex}`(手法別 mean±std/max/p99 [ms]、33.3ms 周期との比)
- 実②表(`exp2_corridor_table`: RPP論文 Table 2 準拠+**制約違反率**)と per-trial 記録
- 図: 経路比較(MPPI 含む)、MPPI 速度プロファイル、**exp2 キー図**(v_cmd + dynamic window 帯 + scan 距離 + 違反マーカー)
- `summary.txt`(入力ファイル数・警告の集約)

部分実行: `--only exp1` / `--only timing` / `--only exp2`。

### レコーダ整合性チェック(初回のみ推奨)

PP系1走行について、GUIレコーダCSVとプラグイン内蔵CSV(コンテナ内
`install/dwpp_test_simulation/share/dwpp_test_simulation/data/dynamic_window_pure_pursuit_log_*.csv`)を突き合わせ:

```bash
python3 scripts/check_recorder_consistency.py \
  --recorder-csv data/real_robot_experiment_revision/exp1_mppi/PathC/DWPP/PathC_DWPP_*.csv \
  --plugin-csv   <上記プラグインCSV>
```
v_cmd/w_cmd 一致(1e-6)と violation 一致率 >99% で PASS。
(注: dw_* はプラグイン側が regulation 適用後のため、regulation 発動区間では一致しない=情報表示のみ)

## 5. 実験日終了時

- `data/real_robot_experiment_revision/` を別ディスク(またはNAS)へコピー。
- 実験ノート: MPPI の std 採用値、AMCL 再初期化タイミング、衝突カウント、バッテリー、天候/人流など。

## 既知の注意点

- `params_path` を指定し忘れると **nelson_test_params.yaml** が読まれる(チェックコマンドで検出)。
- MPPI は unstamped Twist 前提(Jazzy デフォルト)。cmd が流れなければ `enable_stamped_cmd_vel` を疑う。
- レコーダの制約値は **params_path のコントローラブロックから自動同期**(`limits_params_file`)。起動ログで `Recorder limits synced` を確認。同期失敗時は宣言デフォルト(凍結値)にフォールバックし警告が出る(実②では致命的なので必ず確認)。実①の制約は凍結値から変更禁止。
- GUI レコーダの curvature / v_reg 列は常に NaN(コントローラ非依存化のため)。dynamic window スナップショット図が必要な解析はプラグインCSV(PP系のみ)を使う。
