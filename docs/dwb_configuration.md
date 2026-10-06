# MPPI・DWBのECPP実験2からの設定変更

DWBは予測時間8.0 s、Oscillationの解除時間1.0 s、停止とみなす並進速度0.005 m/sを使う。
予測時間と解除時間は前回の採用値を保ち、2026-10-06の続き2の依頼に従って停止速度だけを
0.11 m/sから下げた。0.005 m/sは1周期の各軸の速度増分0.22/30=0.007333 m/sより小さく、
零並進へ1周期で減速できる範囲でRotateToGoalの停止判定を行うための値である。
指定した全体検証の結果と計測値は、原稿側の`docs/hardware/baseline_planner_verification.md`に残す。

MPPIとDWBの出発点は、ECPP論文の実機実験2で使用した
`nav2_error_compensated_pure_pursuit_controller/config/paper_experiment2_params.yaml`
の各ブロックである。2026-10-06に、この作業ツリーの設定を正本と照合した。
正本のSHA-256は`5d4286cb52b18977004057bb92437f455999a0eb6401ccb9d6befa9df1c7faf5`である。
対象はHSRの65試行計画のうち実験2のみで、両手法とも5試行を予定する。
物理HSRの試行は0件であり、以下の完走確認は合成入力による接続試験である。

変更の区分は依頼文と同じく、1: 全方位化、2: HSRと実験の共通設定への適用、
3: インストール済みHumbleとの互換性、4: 合成入力で完走できなかった場合の最小変更、
とする。表の「維持」はECPPと同じ値を表す。`TimedController`は既存の共通計測であり、
今回その実装・計測範囲・手法の割り当てを変更していない。

表は`prepare`が展開する公称値である。速度・加減速度・周期・ゴール許容誤差は
`dwvp_access_experiment.yaml`から展開し、テンプレートに重複して持たせない。
速度は各並進軸±0.22 m/s、回転±0.6 rad/s、公称の加減速度の大きさは
(0.22, 0.22, 0.6)である。加減速度には条件の倍率を掛け、0.5倍では
(0.11, 0.11, 0.3)になる。MPPIとDWBを実験1bに割り当てる変更はしていない。
共通goal checkerの位置0.1 m、姿勢0.3 rad、`stateful=false`も維持する。

## 維持した探索と評価

MPPIの予測は60ステップ×1/30 s=2 s、標本数2000、反復1、温度0.2、gamma 0.015である。
xの標準偏差0.2 m/sと角速度の標準偏差0.4 rad/sはECPPのままとし、yも0.2 m/sにする。
8つのcriticの構成・重み・距離閾値はECPPの値を維持する。
`PathAlignCritic.use_path_orientations=false`も維持する。MPPIは接線姿勢の実験2にのみ使う。
従来の明示値`reset_period=1.0`と`enforce_path_inversion=false`はテンプレートから外し、
同じ値であるインストール済みHumbleの既定値を使う。

DWBの生成器`LimitedAccelGenerator`、周期1/30 s、x/回転の標本数20/20、
7つのcriticと重み、空間・角度の刻みはECPPと同じである。
yの標本数は既存の5を維持する。これで負・零・正の横速度を候補にでき、
20へ増やすことによる候補数の増加を避ける。標本数の指定はx/y/回転で20/5/20である。
この選択の実機での最適性は評価していない。
`max_speed_xy=hypot(0.22,0.22)`は、共通の各軸の箱に余分なノルム制約を加えないための値である。

## Humbleと円形costmapの制約

検証した既存イメージは`docker-hsr:latest`で、MPPIは
`ros-humble-nav2-mppi-controller 1.1.19-1jammy.20251017.040435`、
DWB coreは`1.1.19-1jammy.20251017.040423`である。
コンテナ内のヘッダ、共有ライブラリ、起動後のパラメータ一覧で確認した。
検証時のイメージIDとログの場所は検証記録に残す。

- このMPPIの`ControlConstraints`には速度の4項目だけがあり、`ax_max`、`ax_min`、
  `ay_max`、`ay_min`、`az_max`は使用できない。共通velocity smootherが適用指令の
  加減速度を制限する。MPPI内部で加速度制約を考慮した予測を行うとは主張しない。
- `CostCritic.near_collision_cost`と`CostCritic.trajectory_point_step`は宣言されないため省略する。
  この版の実装を使うもので、新版の同名機能と同等とはみなさない。
- `PathAngleCritic.mode=0`の前進優先は、この版の`forward_preference=true`で指定する。
- `DWB.stateful`は宣言されないため省略する。完了判定には既存の共通goal checkerを使う。
- ECPPの`CostCritic.consider_footprint=true`を最初に試すと、共通costmapが
  `robot_radius=0.22`の円形で、polygonのfootprintを持たないためconfigureに失敗した。
  エラーは`Considering footprint in collision checking but no robot footprint provided in the costmap.`である。
  `false`にすると起動し、両手法の全設定項目が実行時の値と一致した。
  共通costmapやrobot radiusは変えていない。
## MPPIの全パラメータ

| パラメータ | ECPP 実験2 | HSR 公称値 | 理由・区分 |
|---|---|---|---|
| `plugin` | `nav2_mppi_controller::MPPIController` | `dwpp_test_simulation::TimedController` | 2: 既存の共通計測を維持。内側のcontrollerはECPPと同じ |
| `time_steps` | `60` | `60` | 維持 |
| `model_dt` | `0.0333333333` | `0.0333333333` | 維持 |
| `batch_size` | `2000` | `2000` | 維持 |
| `ax_max` | `1.7` | `省略` | 3: このHumble MPPIは未対応。共通smootherで加減速度を制限 |
| `ax_min` | `-1.7` | `省略` | 3: このHumble MPPIは未対応。共通smootherで加減速度を制限 |
| `ay_max` | `0` | `省略` | 3: このHumble MPPIは未対応。共通smootherで加減速度を制限 |
| `ay_min` | `-0` | `省略` | 3: このHumble MPPIは未対応。共通smootherで加減速度を制限 |
| `az_max` | `5.5` | `省略` | 3: このHumble MPPIは未対応。共通smootherで加減速度を制限 |
| `vx_std` | `0.2` | `0.2` | 維持 |
| `vy_std` | `0` | `0.2` | 1: 全方位化。yの標準偏差はECPPのxと同じ |
| `wz_std` | `0.4` | `0.4` | 維持 |
| `vx_max` | `0.5` | `0.22` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `vx_min` | `0` | `-0.22` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `vy_max` | `0` | `0.22` | 1, 2: yを有効化し、共通のxと同じ上限を使う |
| `wz_max` | `1.5` | `0.6` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `iteration_count` | `1` | `1` | 維持 |
| `prune_distance` | `1.7` | `1.7` | 維持 |
| `transform_tolerance` | `0.1` | `0.1` | 維持 |
| `temperature` | `0.2` | `0.2` | 維持 |
| `gamma` | `0.015` | `0.015` | 維持 |
| `motion_model` | `DiffDrive` | `Omni` | 1: 全方位の運動モデルを使う |
| `visualize` | `false` | `false` | 維持 |
| `regenerate_noises` | `false` | `false` | 維持 |
| `TrajectoryVisualizer.trajectory_step` | `5` | `5` | 維持 |
| `TrajectoryVisualizer.time_step` | `3` | `3` | 維持 |
| `critics` | `ConstraintCritic, CostCritic, GoalCritic, GoalAngleCritic, PathAlignCritic, PathFollowCritic, PathAngleCritic, PreferForwardCritic` | `ConstraintCritic, CostCritic, GoalCritic, GoalAngleCritic, PathAlignCritic, PathFollowCritic, PathAngleCritic, PreferForwardCritic` | 維持 |
| `ConstraintCritic.enabled` | `true` | `true` | 維持 |
| `ConstraintCritic.cost_power` | `1` | `1` | 維持 |
| `ConstraintCritic.cost_weight` | `4` | `4` | 維持 |
| `GoalCritic.enabled` | `true` | `true` | 維持 |
| `GoalCritic.cost_power` | `1` | `1` | 維持 |
| `GoalCritic.cost_weight` | `5` | `5` | 維持 |
| `GoalCritic.threshold_to_consider` | `1` | `1` | 維持 |
| `GoalAngleCritic.enabled` | `true` | `true` | 維持 |
| `GoalAngleCritic.cost_power` | `1` | `1` | 維持 |
| `GoalAngleCritic.cost_weight` | `3` | `3` | 維持 |
| `GoalAngleCritic.threshold_to_consider` | `0.5` | `0.5` | 維持 |
| `PreferForwardCritic.enabled` | `true` | `true` | 維持 |
| `PreferForwardCritic.cost_power` | `1` | `1` | 維持 |
| `PreferForwardCritic.cost_weight` | `5` | `5` | 維持 |
| `PreferForwardCritic.threshold_to_consider` | `0.5` | `0.5` | 維持 |
| `CostCritic.enabled` | `true` | `true` | 維持 |
| `CostCritic.cost_power` | `1` | `1` | 維持 |
| `CostCritic.cost_weight` | `3.81` | `3.81` | 維持 |
| `CostCritic.near_collision_cost` | `253` | `省略` | 3: このHumble CostCriticはパラメータを宣言しない |
| `CostCritic.critical_cost` | `300` | `300` | 維持 |
| `CostCritic.consider_footprint` | `true` | `false` | 2: 共通の円形costmapに合わせる。trueではconfigureに失敗 |
| `CostCritic.collision_cost` | `1000000` | `1000000` | 維持 |
| `CostCritic.near_goal_distance` | `1` | `1` | 維持 |
| `CostCritic.trajectory_point_step` | `2` | `省略` | 3: このHumble CostCriticはパラメータを宣言しない |
| `PathAlignCritic.enabled` | `true` | `true` | 維持 |
| `PathAlignCritic.cost_power` | `1` | `1` | 維持 |
| `PathAlignCritic.cost_weight` | `30` | `30` | 維持 |
| `PathAlignCritic.max_path_occupancy_ratio` | `0.05` | `0.05` | 維持 |
| `PathAlignCritic.trajectory_point_step` | `4` | `4` | 維持 |
| `PathAlignCritic.threshold_to_consider` | `0.5` | `0.5` | 維持 |
| `PathAlignCritic.offset_from_furthest` | `13` | `13` | 維持 |
| `PathAlignCritic.use_path_orientations` | `false` | `false` | 維持 |
| `PathFollowCritic.enabled` | `true` | `true` | 維持 |
| `PathFollowCritic.cost_power` | `1` | `1` | 維持 |
| `PathFollowCritic.cost_weight` | `10` | `10` | 維持 |
| `PathFollowCritic.offset_from_furthest` | `13` | `13` | 維持 |
| `PathFollowCritic.threshold_to_consider` | `1` | `1` | 維持 |
| `PathAngleCritic.enabled` | `true` | `true` | 維持 |
| `PathAngleCritic.cost_power` | `1` | `1` | 維持 |
| `PathAngleCritic.cost_weight` | `6` | `6` | 維持 |
| `PathAngleCritic.offset_from_furthest` | `4` | `4` | 維持 |
| `PathAngleCritic.threshold_to_consider` | `0.5` | `0.5` | 維持 |
| `PathAngleCritic.max_angle_to_furthest` | `1` | `1` | 維持 |
| `PathAngleCritic.mode` | `0` | `省略` | 3: mode=0の前進優先をHumbleのforward_preference=trueで指定 |
| `PathAngleCritic.forward_preference` | `省略` | `true` | 3: mode=0の前進優先をHumbleのforward_preference=trueで指定 |
| `wrapped_plugin` | `省略` | `nav2_mppi_controller::MPPIController` | 2: 既存の共通計測を維持。内側のcontrollerはECPPと同じ |

## DWBの全パラメータ

| パラメータ | ECPP 実験2 | HSR 公称値 | 理由・区分 |
|---|---|---|---|
| `plugin` | `dwb_core::DWBLocalPlanner` | `dwpp_test_simulation::TimedController` | 2: 既存の共通計測を維持。内側のcontrollerはECPPと同じ |
| `debug_trajectory_details` | `false` | `false` | 維持 |
| `min_vel_x` | `0` | `-0.22` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `min_vel_y` | `0` | `-0.22` | 1, 2: yを有効化し、共通のxと同じ上限を使う |
| `max_vel_x` | `0.5` | `0.22` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `max_vel_y` | `0` | `0.22` | 1, 2: yを有効化し、共通のxと同じ上限を使う |
| `max_vel_theta` | `1.5` | `0.6` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `min_speed_xy` | `0` | `0` | 維持 |
| `max_speed_xy` | `0.5` | `0.311126984` | 1, 2: 各軸の箱に余分なノルム制約を課さない |
| `min_speed_theta` | `0` | `0` | 維持 |
| `acc_lim_x` | `1.7` | `0.22` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `acc_lim_y` | `0` | `0.22` | 1, 2: yを有効化し、共通のxと同じ上限を使う |
| `acc_lim_theta` | `5.5` | `0.6` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `decel_lim_x` | `-1.7` | `-0.22` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `decel_lim_y` | `0` | `-0.22` | 1, 2: yを有効化し、共通のxと同じ上限を使う |
| `decel_lim_theta` | `-5.5` | `-0.6` | 2: 共通の速度・加減速度制約。加減速度は条件の倍率に従う |
| `vx_samples` | `20` | `20` | 維持 |
| `vy_samples` | `1` | `5` | 1: 既存の5を維持し、負・零・正の横速度を標本化 |
| `vtheta_samples` | `20` | `20` | 維持 |
| `sim_time` | `2` | `8` | 4: 格子幅からの下限6.818 sを切り上げた7 sが終端で失敗し、次の8 sを採用 |
| `linear_granularity` | `0.05` | `0.05` | 維持 |
| `angular_granularity` | `0.025` | `0.025` | 維持 |
| `transform_tolerance` | `0.2` | `0.2` | 維持 |
| `xy_goal_tolerance` | `0.1` | `0.1` | 維持 |
| `trans_stopped_velocity` | `0.25` | `0.005` | 2: 比を保った0.11では早すぎる停止判定で呼出しが失敗したため、停止速度 < 加速度上限×制御周期を満たす値へ変更 |
| `short_circuit_trajectory_evaluation` | `true` | `true` | 維持 |
| `stateful` | `true` | `省略` | 3: Humble DWBは未使用。共通goal checkerのstateful=falseは維持 |
| `critics` | `RotateToGoal, Oscillation, BaseObstacle, GoalAlign, PathAlign, PathDist, GoalDist` | `RotateToGoal, Oscillation, BaseObstacle, GoalAlign, PathAlign, PathDist, GoalDist` | 維持 |
| `Oscillation.oscillation_reset_time` | 省略（既定`-1`） | `1` | 4: 8 sでも終端で停滞したため追加。続き2でも維持 |
| `BaseObstacle.scale` | `0.02` | `0.02` | 維持 |
| `PathAlign.scale` | `32` | `32` | 維持 |
| `PathAlign.forward_point_distance` | `0.1` | `0.1` | 維持 |
| `GoalAlign.scale` | `24` | `24` | 維持 |
| `GoalAlign.forward_point_distance` | `0.1` | `0.1` | 維持 |
| `PathDist.scale` | `32` | `32` | 維持 |
| `GoalDist.scale` | `24` | `24` | 維持 |
| `RotateToGoal.scale` | `32` | `32` | 維持 |
| `RotateToGoal.slowing_factor` | `5` | `5` | 維持 |
| `RotateToGoal.lookahead_time` | `-1` | `-1` | 維持 |
| `trajectory_generator_name` | `dwb_plugins::LimitedAccelGenerator` | `dwb_plugins::LimitedAccelGenerator` | 維持 |
| `sim_period` | `0.0333333333` | `0.0333333333` | 維持 |
| `wrapped_plugin` | `省略` | `dwb_core::DWBLocalPlanner` | 2: 既存の共通計測を維持。内側のcontrollerはECPPと同じ |

## 前回のDWB設定の検証（2.0 sと4.545 s）

すべて同じ合成E2経路（x=0〜1.2 m、y=0.15 sin(πx/1.2)、接線姿勢）と
0.05 mの空き地図を用いた。実機、AMCL、TMC driverは使っていない。
共通smoother、進捗判定、ゴール判定を緩めず、controllerとrecorderを既存の接続試験から再利用した。
各候補は独立したsessionに設定を凍結し、実行時の全設定項目との一致を確認した。

| 順序 | 基準から変えた値 | 結果（中断までの記録時間、最終x） | 採否・理由 |
|---|---|---|---|
| 1 | ECPPに区分1〜3のみ適用。`sim_time=2.0`、`trans_stopped_velocity=0.11` | 失敗、32.827 s、0.000367 m | 前回は基準値として保持。今回の8 sで置き換え |
| 2 | 1から`sim_time=2.0×0.5/0.22=4.545454545`だけ変更 | 失敗、30.010 s、0.000097 m | 距離換算でも発進しないため不採用 |
| 3 | 2から`PathAlign.forward_point_distance: 0.1→0.125`だけ変更 | 失敗、30.001 s、0.000119 m | 格子幅の半分だけ参照点をずらしても進まず、不採用 |
| 4 | 2から`GoalAlign.forward_point_distance: 0.1→0.125`だけ変更（3の変更は外す） | 失敗、36.519 s、0.132529 m | 初期移動後に停滞し、不採用 |
| 5 | 2から`Oscillation.oscillation_reset_time: 既定−1→1.0`だけ変更（4の変更は外す） | 失敗、30.032 s、0.000075 m | 時間による方向制限の解除だけでは進まず、不採用 |
| 6 | 4に`Oscillation.oscillation_reset_time=1.0`だけ追加 | 失敗、36.630 s、0.131670 m | 初期移動後の停滞も解消せず、両変更とも不採用 |

前回の全候補で`Failed to make progress`によるaction status 6を確認した。
criticの構成・重み、標本数、速度・加速度、共通の判定条件は探索で変えていない。
以下の追加検証は2026-10-06の続きの依頼によるものである。

## 格子幅から決める予測時間

`LimitedAccelGenerator`は1周期で到達できる速度を候補とする。静止からの各並進軸の
最大速度増分は、加速度上限aと制御周期Δtの積aΔtである。この速度で予測した変位が
costmapの格子幅r以上になる規則を使うと、予測時間の下限はT_min=r/(aΔt)となる。
公称値では次の計算になる。

- r=0.05 m、a=0.22 m/s²、Δt=1/30 s。
- aΔt=0.007333333 m/s、T_min=0.05/(0.22/30)=6.8181818 s。
- 最初の候補7.0 sではaΔtT=0.051333 m、次の8.0 sでは0.058667 m。
- ECPPの2.0 sでは0.014667 m、前回の距離換算4.545454545 sでは0.033333 mで、
  いずれも各軸の変位が格子幅に届かない。ECPPの加速度1.7 m/s²では2.0 sでも0.113333 mになる。

この規則は、静止からの零速度候補と最大速度増分の候補の変位を、各軸について
格子幅と比較する。すべての隣接標本やcriticの点数を区別できる保証でも、
完走や障害物回避の保証でもない。前回の診断は停滞と格子の関係を示唆したが、
全候補のcriticスコアを照合して停滞の原因を一つに特定したものではない。
ECPPの予測時間からの変更は区分4に当たり、7.0 sでの失敗を確認してから8.0 sを試した。

| 順序 | 前回の採用値からの変更 | 合成E2経路の結果 | 採否・理由 |
|---|---|---|---|
| 7 | `sim_time=7.0`のみ | 44.828 sで中断。最終位置誤差0.099529 m、姿勢誤差0.859642 rad | 発進して位置許容差には入ったが、終端の姿勢合わせが完了せず不採用 |
| 8 | `sim_time=8.0`のみ | 43.298 sで完走。最終位置誤差0.098820 m、姿勢誤差0.299218 rad、平均計算時間5.288 ms | 単独試行では完走したが、全体検証で終端の停滞が再発。この組合せは不採用 |
| 9 | 8に`Oscillation.oscillation_reset_time=1.0`のみ追加 | 単独試行は35.626 s、全体検証内は35.779 sで完走。平均計算時間は2.860/2.868 ms | 完走を確認して保持。ただしcontroller呼出しが1回失敗し、全体検証はexit 1 |

7と8では停止速度0.11 m/sとOscillationの解除時間の既定値−1を保持した。
8の全体検証ではaction status 6で中断し、最終位置誤差0.099638 m、姿勢誤差0.844739 radだった。
この失敗を確認してから9を試した。単独の完走を、2回の全体検証の成功と同一には扱わない。
9でも停止速度0.11 m/sを保持したが、単独試行の計測には失敗したcontroller呼出しが1回含まれた。
計算時間は既存TimedControllerの計測値であり、計測範囲や合格条件は変更していない。

前回は全体検証の`timing['failed_calls']==0`が満たされず、指定コマンドの2回の成功は
未達だった。失敗した呼出しの付近では、単独試行のodomのx速度は約0.007719 m/s、
並進ノルムは約0.008546 m/sだった。x速度は1周期の減速量0.007333 m/sを上回る一方、
ノルムは停止判定の0.11 m/sより小さい。イメージ内の
`/opt/ros/humble/include/dwb_critics/rotate_to_goal.hpp`によれば、RotateToGoalは
十分に停止したとみなした最終段階で並進を含む候補を拒否する。
この仕様と速度の記録は、零並進へ1周期で到達できない段階で回転だけを要求したという
説明と整合する。ただし、この呼出しの全criticの個別スコアは収集していない。

## 1周期の速度増分より小さい停止判定（続き2）

続き2の依頼では、停止速度の引下げをHSRへの適用に必要な変更（区分2）として明示的に認めた。
ECPPの上限比0.25/0.5を保つ換算値0.11 m/sは使わず、次の規則で0.005 m/sを採用する。

- 停止とみなす並進速度 < 加速度の上限 × 制御周期。
- 公称の各並進軸では0.005 < 0.22×(1/30)=0.007333333 m/s。
- 並進速度のノルムが0.005 m/s未満なら各軸の絶対値もその値未満となり、
  対称な加減速度制約の下で1周期以内に零速度へ減速できる。

`LimitedAccelGenerator`は1周期で到達できる速度だけを候補にするため、
停止判定後にRotateToGoalが求める零並進が候補に含まれるようにする。
これはこの生成器と公称の制約に対する設定規則であり、実機の停止精度や完走を保証するものではない。
予測時間8.0 s、Oscillationの解除時間1.0 s、ECPP由来のcritic構成・重み・標本数、
MPPIの設定、TimedControllerと試験の合格条件は変更しない。

指定された二つの検証コマンドは各2回ともexit 0で完了した。DWBの合成E2結果は次のとおりである。

| 全体検証 | 走行時間 [s] | 平均計算時間 [ms/呼出し] | 失敗した呼出し [回] |
|---|---:|---:|---:|
| 続き2・1回目 | 43.145 | 5.365 | 0 |
| 続き2・2回目 | 43.147 | 5.255 | 0 |

## 加速度倍率との関係

加速度を公称値のk倍にすると、この規則の下限はT_min(k)=6.8181818/k sとなる。
0.5倍では13.6363636 sで、1秒単位に切り上げる最初の候補は14.0 sとなる。
8.0 sのままではaΔtT=0.029333 mで、0.05 mの格子幅に届かない。
したがって、将来DWBを0.5倍の条件へ割り当てるなら、加速度倍率に応じて
予測時間を再計算し、完走と計算時間を改めて確認する必要がある。
停止速度についても、0.5倍では各軸の増分が0.003666667 m/sとなり、
今回の0.005 m/sは規則を満たさないため、同時に見直す必要がある。

現在のDWBの割り当ては公称の実験2だけなので、条件に応じた予測時間の自動変更は実装しない。
0.5倍のパラメータファイルにも共通テンプレートの8.0 sは含まれるが、
割り当て外のDWBを実行できる設定として検証したものではない。
手法の割り当てと既存の拒否処理は維持する。
