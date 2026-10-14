#!/usr/bin/env python3
"""Export the fixed P2v2 protocol, figures and complete sweep/verification tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np
import dwvp_access_e2_report as base


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def sweep_table(rows, controller):
    if controller == 'DWB':
        columns = ['sim_time [s]', 'yaw scale', 'position scale']
        values = lambda r: [r['overrides'][k] for k in ('sim_time', 'PathOrientationCritic.scale', 'PathPositionCritic.scale')]
    else:
        columns = ['Angle weight', 'Align weight', 'wz_std']
        values = lambda r: [r['overrides']['PathAngleCritic']['cost_weight'], r['overrides']['PathAlignCritic']['cost_weight'], r['overrides']['wz_std']]
    head = columns + ['完走', '位置積分 [m s]', '方位積分 [deg s]', '選定得点', '直線巡航 [m/s]', '0.22 未満']
    lines = ['| '+' | '.join(head)+' |', '| '+' | '.join(['---']*len(head))+' |']
    for r in rows:
        if r['controller'] != controller:
            continue
        data = [str(x) for x in values(r)] + ['成功' if r['success'] else '失敗']
        data += [number(r.get(k), 4 if 'position' in k else 3) for k in
                 ('eval_position_error_integral_m_s', 'eval_heading_error_integral_deg_s', 'selection_score', 'straight_cruise_mean_speed_m_s')]
        data += ['はい' if r.get('straight_cruise_below_022') else 'いいえ' if r.get('straight_cruise_below_022') is False else '—']
        lines.append('| '+' | '.join(data)+' |')
    return '\n'.join(lines)


def horizon_summary(rows):
    output=[]
    for duration in (1.7, 2., 3., 8.):
        samples=[r for r in rows if r['controller']=='DWB' and r['overrides']['sim_time']==duration]
        completed=[r for r in samples if r['success'] and r.get('selection_score') is not None]
        best=min(completed,key=lambda r:r['selection_score']) if completed else None
        speeds=[r['straight_cruise_mean_speed_m_s'] for r in samples if r.get('straight_cruise_mean_speed_m_s') is not None]
        output.append(dict(sim_time_s=duration, completed=len(completed), candidates=len(samples),
                           observed_speed_count=len(speeds),
                           all_observed_speed_mean_m_s=float(np.mean(speeds)) if speeds else None,
                           all_observed_speed_range_m_s=[min(speeds),max(speeds)] if speeds else None,
                           best=best))
    return output


def horizon_table(summary):
    lines=['| sim_time [s] | 完走 | 全候補の観測巡航平均 [m/s] | 完走候補の最小得点設定 yaw / position | 巡航 [m/s] | 位置平均 [m] | 方位平均 [deg] |',
           '|---:|---:|---:|---|---:|---:|---:|']
    for row in summary:
        best=row['best']
        settings=f"{best['overrides']['PathOrientationCritic.scale']:g} / {best['overrides']['PathPositionCritic.scale']:g}" if best else 'なし'
        lines.append('| '+' | '.join([str(row['sim_time_s']),f"{row['completed']}/{row['candidates']}",
            number(row['all_observed_speed_mean_m_s'])+f" (n={row['observed_speed_count']})", settings,
            *[number(best.get(k) if best else None,4 if 'position' in k else 3) for k in
              ('straight_cruise_mean_speed_m_s','eval_mean_position_error_m','eval_mean_heading_error_deg')]])+' |')
    return '\n'.join(lines)


def result_table(groups):
    headers=['手法','成功','位置平均 [m]','位置最大 [m]','方位平均 [deg]','方位最大 [deg]','走行時間 [s]','計算時間 [ms]','直線巡航 [m/s]']
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*8)+' |']
    for g in groups:
        row=[base.LABELS[g['method']], f"{g['successes']}/{g['repeats']}"]
        row += [base.format_stat(g[k], 4 if 'position' in k else 2 if 'heading' in k or k=='travel_time_s' else 3) for k in base.METRICS]
        lines.append('| '+' | '.join(row)+' |')
    return '\n'.join(lines)


def horizon_figure(rows, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ['straight_cruise_mean_speed_m_s', 'eval_mean_position_error_m', 'eval_mean_heading_error_deg']
    labels = ['Straight cruise [m/s]', 'Mean position error [m]', 'Mean heading error [deg]']
    for axis, metric, label in zip(axes, metrics, labels):
        for row in rows:
            if row['controller'] != 'DWB' or row.get(metric) is None:
                continue
            if not row['success'] and metric != metrics[0]:
                continue
            axis.scatter([1.7, 2., 3., 8.].index(row['overrides']['sim_time']), row[metric],
                         c='#0072B2' if row['success'] else '#999999',
                         marker='o' if row['success'] else 'x', alpha=.7)
        axis.set(xlabel='DWB sim_time [s]', ylabel=label, xticks=[0,1,2,3],
                 xticklabels=['1.7','2.0','3.0','8.0'], xlim=(-.4,3.4))
        axis.grid(alpha=.25)
    axes[0].axhline(.22, ls=':', c='black')
    fig.suptitle('Blue: completed; gray x: incomplete partial cruise; errors: completed only')
    fig.tight_layout();fig.savefig(output/'dwb_horizon_comparison.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();w=a.work;o=a.output
    o.mkdir(parents=True,exist_ok=False)
    base.METHODS=['VP_CLIP','DWVP','MPPI','DWB']
    base.LABELS['DWVP']='DWVP (L=0.11 m)'
    base.METRICS += ['straight_cruise_mean_speed_m_s']
    facts=base.export_route('P2v2',w/'P2v2',w/'P2v2_final_check',w/'P2v2_sweep_final',w/'P2v2_verify',w,o)
    groups=facts['groups'];r=facts['checks'];m=facts['metadata']
    sweep=base.read_json(w/'P2v2_sweep_final/report.json')
    horizons=horizon_summary(sweep['rows'])
    horizon_figure(sweep['rows'], o)
    (o/'dwb_horizon_summary.json').write_text(json.dumps(horizons,indent=2,allow_nan=False)+'\n')
    (o/'P2v2/results_table.md').write_text(result_table(groups)+'\n')
    shutil.copy2(w/'P2v2_sweep_final/sweep.csv',o/'sweep.csv')
    shutil.copy2(w/'P2v2_sweep_final/plan_before_results.json',o/'plan_before_results.json')
    shutil.copy2(w/'P2v2_selected.yaml',o/'selected_config.yaml')
    for name in ('build-final.log','critic-test-final.log','tooling-test-final.log','runtime-provenance.txt','validation.json','reproduce.sh','source_manifest.json'):
        if (w/name).exists():shutil.copy2(w/name,o/name)
    shutil.copytree(w/'nav2_source_evidence',o/'nav2_source_evidence')
    with tarfile.open(o/'development_attempts.tar.gz','w:gz') as archive:
        for name in ('P2v2_shift010','P2v2_forward','P2v2_clean','P2v2_minimal_shift','P2v2_sweep'):
            if (w/name).exists():archive.add(w/name,arcname=name)
        for name in ('P2v2_initial','P2v2_shift005'):
            previous=Path('/tmp/dwvp-e2-design-20261014')/name
            if previous.exists():archive.add(previous,arcname=name)
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source.parent/'dwb_path_tracking_critics',o/'implementation/dwb_path_tracking_critics')
    for rel in ('scripts/dwvp_access_path.py','scripts/dwvp_access_e2.py','scripts/dwvp_access_experiment.py',
                'scripts/dwvp_access_e2_report.py','scripts/dwvp_access_e2_v2_report.py','scripts/e2_tooling.sh',
                'tests/test_access_e2.py','tests/test_access_path.py','params/dwvp_access_e2_route_design_v2.yaml',
                'params/dwvp_access_path_v2.yaml','package.xml'):
        dest=o/'implementation/dwpp_test_simulation'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,dest)
    clear=r['clearance'];start=m['start'];facing=m['facing']['intervals'][0]
    path = np.loadtxt(w/'P2v2/E2_environment.csv', delimiter=',', skiprows=1)
    requested_difference=math.atan2(math.sin(start[2]-math.pi),math.cos(start[2]-math.pi))
    chosen=json.dumps(sweep['selected'],indent=2,ensure_ascii=False)
    diagnostics='\n'.join(
        f"- {base.LABELS[g['method']]}: 評価区間完了 {g['evaluation_complete']}/{g['repeats']}、data_errors あり {g['data_error_runs']} 試行、plugin 呼び出し失敗 {g['failed_compute_calls']} 回。"
        for g in groups)
    incomplete=[f"{base.LABELS[g['method']]} {g['successes']}/{g['repeats']}" for g in groups if g['successes']!=g['repeats']]
    verification_note=('選定後も全反復で完走した。' if not incomplete else
                       '選定後の完走数は '+ '、'.join(incomplete)+' にとどまった。掃引1回の完走は反復の安定性を保証せず、未完走行の精度値を完走精度として比較することはできない。')
    text=f'''# 実験 2 経路 v2・連続 DWB critic・合成比較（2026-10-14）

ConstrainedSmoother の `minimum_turning_radius=0.3 m` と他の既定値で P2v2 を生成し、地図の円形 footprint 検査を通過した。ただし、実測 XY 最小半径は {r['xy_min_radius_m']:.5f} m で、目標の 0.25〜0.35 m は未達である。以下はこの生成結果に対する実 Humble plugin と free-map fake robot の合成実験であり、半径 0.3 m の丸いコーナーを実現した実験や実機の結果としては扱えない。

## 経路の生成と検査

地図は原稿リポジトリの `map/map.yaml` と `map.pgm` を読み取り専用で使用した。+y が部屋の手前、−y が窓側、左が棚、右が TV。著者確認により天板の張り出しはなく、占有点には椅子の脚も含む。実験時は椅子を地図取得時の位置に固定する。clearance は壁・棚・机と椅子の脚などの占有セル矩形までの中心距離であり、実機の測位誤差は含まない。

要求 start (2.5, 0.05, π)、via (−0.15, 0.05)、(−0.15, −2.5)、goal (2.5, −2.6, 0) から開始した。上側 via は公開 cost=99（inscribed）、占有セルまで約 0.175 m で半径 0.22 m の機体に不適切だった。x を +0.05 m 移した (−0.10, 0.05) は via 自体が非致死セルとなったが、平滑化後の衝突検査に失敗した。試した 0.05 m 刻みの候補で最初に通った **(−0.05, 0.05)**（+0.10 m）を採用し、下側 via は変更していない。これが連続座標上の厳密な最小移動量であるとは主張しない。

3 区間の NavFn 経路を連結し、既存のセル内反転点除去を適用してから、一度だけ ConstrainedSmoother に渡した。連結点の重複除去で生じる新しい反転点を再検査する不具合修正を含む。NavFn がセルへ丸めた始点は要求座標へ戻し、入力方位は各次線分の向き（端点は要求 yaw）とした。これにより前進経路の方位表現による反転判定を避ける。最適化パラメータは旋回半径以外を変更していない。SimpleSmoother への代替は行っていない。

その後、最大 0.02 m で XY を再標本化し、従来と同じ s ± 0.10 m の差分で接線 yaw を推定した。指定線分 (−0.15, −0.8)–(−0.15, −1.9)、yaw=π、遷移長 0.35 m の正対プロファイルを適用した。初期姿勢には CSV 最初の pose を用いた。

| 項目 | 結果 |
|---|---|
| 経路長 / 点数 | {r['length_m']:.4f} m / {r['points']} |
| 実測 XY R_min | {r['xy_min_radius_m']:.5f} m（目標未達） |
| 角速度超過点（接線 / 正対込み） | {r['tangent_yaw']['excess_count']} / {r['excess_count']} |
| 最大要求角速度（正対込み） | {r['max_abs_required_yaw_rate_rad_s']:.4f} rad/s |
| 最小 clearance / 円形外周余裕 | {clear['min_m']:.4f} / {clear['min_m']-.22:.4f} m |
| 最小位置 / 弧長 | ({clear['position'][0]:.4f}, {clear['position'][1]:.4f}) / {clear['progress_m']:.4f} m |
| 開始 yaw / CSV 最初の yaw との差 | {start[2]:.8f} rad / {m['start_yaw_minus_csv_yaw_rad']:.8f} rad |
| 要求 π との差 | {requested_difference:.8f} rad（生成接線に合わせて変更） |
| CSV 終端 yaw / 要求 goal yaw=0 との差 | {path[-1,2]:.8f} / {path[-1,2]:.8f} rad（接線付与で要求 yaw を上書き） |
| CSV の始点 x / 終点 x | {path[0,0]:.8f} / {path[-1,0]:.8f} m |
| 正対区間 | s=[{facing['start_m']:.4f}, {facing['end_m']:.4f}] m |
| 遷移を含む区間 | s=[{facing['transition_start_m']:.4f}, {facing['transition_end_m']:.4f}] m |

角速度検査は速度 0.22 m/s、上限 ±0.6 rad/s。全超過点の座標と要求値は `P2v2/check/report.json`、全標本は `profile.csv`。XY 半径は隣接線分の向き変化を平均線分長で割った離散曲率の最大絶対値の逆数である。元地図の占有・未知セル・地図外に対し、標本間も細分した円形 footprint 検査を行った。inflation radius=0.70 m、cost_scaling_factor=3.0、robot_radius=0.22 m は維持した。

ConstrainedSmoother の曲率項は閾値超過へのペナルティであり、厳密な不等式制約ではない。既定 optimizer は最大 100 反復で、action の完了は曲率目標の達成を意味しない。このデータから未達の原因を一つに特定することはできない。0.35 m の 90° smoothstep 方位遷移も、理想直線上で最大約 1.48 rad/s を要求するため、曲率目標とは別に減速が必要である。

## DWB critic と MPPI との対応

`third_party/dwb_path_tracking_critics` に `PathOrientationCritic` と `PathPositionCritic` を新設した。候補姿勢 p を DWB の `prepare()` が受け取る座標変換・切り詰め済み global plan の各線分へ射影し、最短距離の線分を選ぶ。射影係数は [0,1] に制限し、同距離は経路順とする。参照 yaw は線分両端の最短角差で補間する。

- 位置 raw score: 各姿勢の射影点までの Euclidean 距離 [m]。
- 方位 raw score: 各姿勢と射影した参照 yaw の wrap した絶対角差 [rad]。
- `aggregation: mean` は生成された全姿勢の算術平均、`last` は最後の姿勢だけを採点する。DWB が raw score に `scale` を一度掛ける。両方とも今回は mean。

Humble 1.1.19 の MPPI PathAngleCritic も軌跡姿勢全体の平均を用いる。ただし採点対象は先の経路点への方向であり、射影点の経路 yaw とは異なる。PathAlignCritic の `use_path_orientations=true` は、軌跡と経路の累積距離に基づく対応点に対し、XY と yaw を組み合わせた距離を標本間引き後に平均する。今回は DWB にも参照 yaw を与えたが、目的関数・対応点・ゲート・局所経路範囲を同一にしたわけではない。ネットワークは使用せず、ローカル Git にあった Nav2 tag 1.1.19 の該当ソースと commit を `nav2_source_evidence/` に保存した。

DWB は Oscillation、BaseObstacle、ObstacleFootprint、GoalDist、GoalAlign、RotateToGoal と新しい2 critic を使用する。PathDist と PathAlign は1周期の速度窓内で格子が候補を区別しにくいため外し、PreferForward と Twirling は指定どおり入れない。生成器は LimitedAccelGenerator、周期 1/30 s、標本数 20/5/20、max_speed_xy=0.22 m/s。残した GoalDist・GoalAlign には格子量子化があるため、新 critic の連続性だけで前進や目標速度を保証しない。

## 事前固定した掃引と選定

`plan_before_results.json` を試行前に保存した。DWB は sim_time=1.7/2/3/8 s、方位 scale=1/4/16、位置 scale=4/16/64 の36条件。MPPI は Angle weight=6/12/20、Align weight=15/30、wz_std=0.2/0.4 の12条件、60 steps × 1/30 s=2 s。他の MPPI 条件は既存 E2 設定を継承した。小・中・大は初回の走行結果を見る前に上記の数値で固定し、結果に応じた格子追加や変更はしていない。

各候補は1試行。同じ controller の完走候補に限り、位置誤差時間積分 / 完走候補の最小位置積分 + 方位誤差時間積分 / 完走候補の最小方位積分を最小化した。同点は事前の格子順、最小値 0 は既存の極限規則とした。共通評価区間は s=0 から末尾の 0.6 m 手前。失敗候補の途中までの誤差は選定に使わない。

巡航速度は odometry の並進速度標本の算術平均。共通区間 s∈[0.3, length−0.6] m のうち、参照 XY 接線の弧長微分の絶対値 ≤0.1 rad/m の直線部分に射影された標本を使う。正対の方位遷移は XY が直線なら含む。0.22 未満は丸め前の平均を厳密に比較したもので、付帯 JSON には ±5% 帯内の標本割合も残した。失敗候補は観測できた部分のみで、完走巡航の値ではない。

掃引はネットワークなしの1コンテナ内で独立 ROS_DOMAIN_ID を使い、最大4プロセスで実行した。最終 verify は全試行を順次実行した。初回掃引は削除した PathAlign の不要設定が凍結 YAML に残って実行時設定検証が失敗したため中止した。設定生成だけを修正し、同じ選定規則・格子で全48条件を再実行した。中止した記録は `development_attempts.tar.gz` に保持し、選定に混ぜていない。

### DWB 全36条件

{sweep_table(sweep['rows'],'DWB')}

### MPPI 全12条件

{sweep_table(sweep['rows'],'MPPI')}

### 選定した設定

```json
{chosen}
```

全条件を含む選定済み YAML は `selected_config.yaml`。各試行の controller_server と velocity_smoother の実行時パラメータを保存し、凍結設定との一致を検証した。

### DWB の予測時間ごとの比較

{horizon_table(horizons)}

全候補の観測巡航平均は有限値が得られた n 条件の標本平均をさらに平均した参考値で、未完走候補を含む。右側は全 DWB 格子で正規化した選定得点が、その sim_time の完走候補中で最小の条件。各予測時間で別の格子を追加していない。`dwb_horizon_comparison.png` に全候補の観測巡航と完走候補の誤差を示す。

元地図の障害物は fake robot に配信していないため、この結果だけで減速を机・椅子の inflation cost に帰属させることはできない。一方、共通 local costmap は 3×3 m で、0.22 m/s を8秒外挿すると中心から1.76 mに達する。地図外判定と局所経路の切り詰めも長い予測の候補評価に関わり得る。各 critic を外す検証や costmap 寸法の変更は今回の固定条件外なので行っていない。

## 選定後の4手法・各3反復

VP_CLIP は適応前方注視 0.75 s、0.11〜0.33 m。DWVP は横方向時定数 0.6 s、固定 L=0.11 m。PP 系の cost regulation と rotate-to-heading はオフ。MPPI と DWB は上記の選定値を固定した。

{result_table(groups)}

平均 ± SD は各反復で求めた指標の算術平均 ± 標本標準偏差（ddof=1）。位置と方位の平均誤差は共通評価窓の台形時間積分を時間で割った値。最大欄も試行ごとの最大値を3反復で集計した。走行時間は停止確認まで、計算時間は TimedController による plugin 呼び出しの平均時間でありサーバ全体の周期ではない。失敗の途中までの有限な指標は保持したため、成功数が3/3でない行は完走精度として比較できない。

{diagnostics}

{verification_note}

図中の 0.22 m/s は巡航目標である。共通の速度制限は各並進軸 ±0.22 m/s で、DWB の候補速度にはさらに `max_speed_xy=0.22` の並進ノルム制限がある。MPPI は既存設定を継承しており、斜行時には観測ノルムが約0.311 m/sに達した。全手法に同じ並進ノルム上限を追加した比較ではない。

合成軌跡を元地図に重ねた各反復の最小外周余裕は `P2v2/tracking_map_clearance.json`。free-map plant の controller が元地図の障害物を観測した結果ではない。

## 保存物・検証・再実行

変更ファイルは本パッケージの `package.xml`、`scripts/dwvp_access_path.py`、`scripts/dwvp_access_e2.py`、`scripts/dwvp_access_experiment.py`、`scripts/dwvp_access_e2_report.py`、`scripts/e2_tooling.sh`、`tests/test_access_path.py`、`tests/test_access_e2.py`。新設は `params/dwvp_access_path_v2.yaml`、`params/dwvp_access_e2_route_design_v2.yaml`、`scripts/dwvp_access_e2_v2_report.py` と、隣接する `dwb_path_tracking_critics/` の CMakeLists.txt、package.xml、critics.xml、ヘッダ、実装、テスト、README、LICENSE。本報告ディレクトリ以外の原稿リポジトリには書き込んでいない。

- `P2v2/map_tracking.png`: 地図、経路、正対・遷移区間、4手法の全3反復の軌跡。
- `P2v2/heading_speed.png`: 弧長に対する方位誤差と odometry 並進速度。
- `P2v2/check/path_check.png`: 方位、clearance、曲率、要求角速度の4パネル。
- `P2v2/route/`: 経路 CSV、metadata、NavFn 各区間、smoother 入出力、runtime パラメータ、地図の写し、生成ログ。
- `P2v2/sweep_raw.tar.gz` と `verify_raw.tar.gz`: 全試行の CSV、パラメータ snapshot、timing、判定ログを含む生データ。元の保存先は `{w}`。
- `sweep.csv`、`dwb_horizon_summary.json`、`P2v2/statistics.json`: 機械可読の集計。
- `implementation/`: 今回の実装の写し。`source_manifest.json` は依存コードも含む SHA256、`SHA256SUMS.json` は報告物のハッシュ。
- `reproduce.sh`: 新しい `/tmp` 作業場所で build → path → check-path → sweep → 選定 → verify → tests → report を再実行する。

テスト結果と E1 不変検査は `validation.json`、`critic-test-final.log`、`tooling-test-final.log`。既存の基準 YAML は変更しておらず、全 E1 条件の展開済み Nav2 パラメータも元設定との同一性をテストした。commit、push、branch 切替、外部ネットワーク通信、物理ロボット接続は行っていない。

## 未達・未確定事項

半径目標は未達であり、この経路を半径 0.3 m の丸め経路として採用する判断はできない。optimizer 等の追加調整、別の前処理、または別の smoother を使う場合は、今回は指定外となるため別条件として経路生成と掃引をやり直す必要がある。要求 π と開始接線も一致しなかったため、初期 yaw を CSV に合わせた。開始 yaw と CSV の差はゼロである。終端 yaw も接線付与により要求 0 から {path[-1,2]:.4f} rad に変わった。始点・終点の x はともに 2.5 m で一致している。

最小外周余裕は地図上で約 {clear['min_m']-.22:.3f} m に限られる。実機の通行可否、位置推定誤差、机・椅子配置の再現、乗員・周囲の人の不快感は未評価である。速度・方位の制約設定や合成完走は、これらの検証を代替しない。掃引は各1試行で、MPPI の乱数と実時間スケジューリングによる変動を含む。3反復の SD は今回の条件内のばらつきであり、一般化性能や有意差を示すものではない。
'''
    (o/'README.md').write_text(text)
    hashes={str(f.relative_to(o)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(o.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.json'}
    (o/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(result_table(groups));print(horizon_table(horizons))


if __name__=='__main__':main()
