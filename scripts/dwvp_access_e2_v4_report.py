#!/usr/bin/env python3
"""Export recorded route-v4 results; no ROS, Docker, or network operations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
import yaml

import dwvp_access_e2_report as base
from dwvp_access_e2_v2_report import result_table, sweep_table
from dwvp_access_experiment import render_parameters
from dwvp_access_path import load_map, clearance_profile

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path('/home/decwest/decwest_workspace/tex_docker_environment/projects/DWVP_ACCESS')


def save(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sides(route):
    path = np.loadtxt(route/'E2_environment.csv', delimiter=',', skiprows=1)
    info, pixels, _ = load_map(route/'map.yaml')
    occupied = np.flipud((pixels/255 if info.get('negate',0) else (255-pixels)/255) > info['occupied_thresh'])
    if info['origin'][2] != 0:
        raise ValueError('Side masks are defined in this unrotated laboratory map')
    x = info['origin'][0]+(np.arange(pixels.shape[1])+.5)*info['resolution']
    y = info['origin'][1]+(np.arange(pixels.shape[0])+.5)*info['resolution']
    output = dict(definition='Occupied cell rectangles with centres in y=[-2.6,0.05]; shelf side x<-0.05, desk side x>=-0.05. Distances are from the reference centre, not the robot perimeter or tabletop.')
    for name, half in [('shelf',x<-.05),('desk',x>=-.05)]:
        mask = occupied & half[None,:] & (y[:,None]>=-2.6) & (y[:,None]<=.05)
        distances, cells = clearance_profile(path, info, mask)
        i = int(distances.argmin())
        output[name] = dict(min_m=float(distances[i]), position=path[i,:2].tolist(),
                           nearest_cell=cells[i].tolist(), footprint_margin_m=float(distances[i]-.22))
    return output


def stops(rows, work):
    out = []
    for row in rows:
        if row['success']:
            continue
        folder = base.host_path(row['session'],work)/'runs'/row['trial_id']
        track = base.read_csv(folder/'tracking.csv')
        valid = np.isfinite(np.c_[track['x'],track['y'],track['yaw']]).all(axis=1)
        track = track[valid]
        out.append(dict(trial=row['trial_id'], position=[float(track['x'][-1]),float(track['y'][-1])],
                        duration_s=row.get('duration_s'), goal_distance_m=row.get('final_position_error_m'),
                        evaluation_complete=row.get('evaluation_complete')))
    return out


def load_evidence(folder, rows, work):
    """Publication measurements and native warnings, with their limits explicit."""
    log = (folder/'stack/controller_server.log').read_text()
    warnings = [line for line in log.splitlines() if '[WARN]' in line or '[ERROR]' in line]
    observations = base.read_csv(folder/'costmap_observations.csv')
    periods, ages, compute = [], [], {}
    for row in rows:
        run = base.host_path(row['session'],work)/'runs'/row['trial_id']
        track = base.read_csv(run/'tracking.csv')
        sample = observations[(observations['receive_stamp_s']>=track['stamp_s'][0]) &
                              (observations['receive_stamp_s']<=track['stamp_s'][-1])]
        periods.extend(np.diff(sample['receive_stamp_s']).tolist())
        ages.extend(sample['age_s'][sample['update_stamp_s']>0].tolist())
        timing = base.read_csv(run/'timing.csv')
        compute.setdefault(row['controller'],[]).extend(timing['compute_time_ms'].tolist())
    def stats(values):
        v = np.asarray(values); v = v[np.isfinite(v)]
        return dict(n=len(v), mean=float(v.mean()) if len(v) else None,
                    p95=float(np.percentile(v,95)) if len(v) else None,
                    maximum=float(v.max()) if len(v) else None)
    return dict(update_frequency_configured_hz=5., publish_frequency_configured_hz=2.,
        publication_interval_s=stats(periods), valid_update_stamp_age_s=stats(ages),
        update_period_directly_observed=False,
        update_stamp_note='Humble costmap_raw metadata.update_time is zero in these runs; no update age can be inferred from it.',
        resolution_values_m=sorted(set(observations['resolution_m'].tolist())),
        cell_dimensions=sorted({(int(a),int(b)) for a,b in zip(observations['width'],observations['height'])}),
        update_loop_missed_warnings=len(re.findall(r'Map update loop missed',log,re.I)),
        control_loop_missed_warnings=len(re.findall(r'Control loop missed',log,re.I)),
        sensor_origin_outside_warnings=len(re.findall(r'Sensor origin .*out of map bounds',log)),
        warnings_and_errors=warnings,
        plugin_calls={name:dict(**stats(values), over_33p333_ms=sum(v>1000/30 for v in values)) for name,values in compute.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); w,o = args.work,args.output
    o.mkdir(parents=True,exist_ok=False)
    base.METHODS = ['VP_CLIP','DWVP','MPPI','DWB']
    base.LABELS['DWVP'] = 'EC-DWVP'
    if 'straight_cruise_mean_speed_m_s' not in base.METRICS:
        base.METRICS.append('straight_cruise_mean_speed_m_s')
    facts = base.export_route('B_graph',w/'route_001',w/'check_001',w/'sweep',w/'verify',w,o)
    final_table = result_table(facts['groups'])
    (o/'B_graph/results_table.md').write_text(final_table+'\n')
    shutil.copytree(w/'route_005',o/'inspection_005/route')
    shutil.copytree(w/'check_005',o/'inspection_005/check')
    comparison = dict(v4=sides(w/'route_001'),
        v3=sides(PAPER/'docs/hardware/e2_design_20261014_v3/B_graph/route'))
    save(o/'clearance_comparison.json',comparison)
    resolution, loads = {}, {}
    for label in ('A','B','C'):
        folder = w/('resolution_'+label)
        if not (folder/'report.json').exists():
            continue
        rows = base.read_json(folder/'report.json')['rows']
        base.METHODS = ['DWB']
        resolution[label] = dict(group=base.statistics(rows)[0],stops=stops(rows,w))
        loads[label] = load_evidence(folder/'controllers',rows,w)
        shutil.copytree(folder,o/('resolution_'+label))
    base.METHODS = ['VP_CLIP','DWVP','MPPI','DWB']
    final_rows = base.read_json(w/'verify/report.json')['rows']
    loads['final'] = load_evidence(w/'verify/controllers',final_rows,w)
    save(o/'resolution_statistics.json',resolution)
    save(o/'costmap_and_compute.json',loads)
    sweep = base.read_json(w/'sweep/report.json')
    sweep_md = sweep_table(sweep['rows'],'DWB')
    (o/'sweep_table.md').write_text(sweep_md+'\n')
    for name in ('sweep.csv','plan_before_results.json'):
        shutil.copy2(w/'sweep'/name,o/name)
    diagnosis = None
    if (w/'diagnostic_B').exists():
        shutil.copytree(w/'diagnostic_B',o/'diagnostic_B')
        file = w/'diagnostic_B/controllers/dwb_evaluation.jsonl'
        records = [json.loads(line) for line in file.read_text().splitlines()]
        diagnosis = dict(captured_evaluations=len(records),
            positions=[r['pose'] for r in records],
            goal_dist_ranges=[r['critic_candidate_ranges'].get('GoalDist') for r in records],
            best_velocities=[r['best_velocity'] for r in records],
            rejections=[r['rejected_by_critic'] for r in records],
            first_best_scores=records[0]['best_scores'] if records else None,
            first_straight_candidates=records[0]['straight_candidates'] if records else [],
            source='diagnostic_B/controllers/dwb_evaluation.jsonl',excluded_from_selection_and_timing=True)
        save(o/'dwb_diagnosis.json',diagnosis)
    for file in w.glob('*.log'):
        shutil.copy2(file,o/file.name)
    for name in ('initial_source_hashes.json','initial_E1_rendered.json','geometry_decision.json',
                 'selected_config.yaml','protocol_before_sweep.json','commands.jsonl'):
        if (w/name).exists(): shutil.copy2(w/name,o/name)
    initial = base.read_json(w/'initial_source_hashes.json')
    original = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v3.yaml').read_text())
    config = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v4.yaml').read_text())
    params = ROOT/'params/hsrb_dwvp_access_params.yaml'
    saved_e1 = base.read_json(w/'initial_E1_rendered.json')
    rendered = {c:render_parameters(params,config,c) for c in saved_e1}
    files = [p for folder in ('scripts','params','tests','src') for p in (ROOT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    changes = dict(changed=[p for p in initial if p in hashes and hashes[p]!=initial[p]],
                   new=[p for p in hashes if p not in initial])
    for relative in changes['changed']+changes['new']:
        dest=o/'implementation'/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest)
    save(o/'changed_files.json',changes); save(o/'source_manifest.json',hashes)
    test_log=(w/'tooling-test.log').read_text()
    previous_selected = yaml.safe_load((PAPER/'docs/hardware/e2_design_20261014_v3/selected_config.yaml').read_text())
    previous_controllers = previous_selected['conditions']['E2_environment']['controller_overrides']
    for key in ('vx_samples','vy_samples','vtheta_samples'):
        previous_controllers['DWB'][key] = 7
    validation=dict(E1_rendered_equal_to_initial=rendered==saved_e1,
        E1_config_equal_to_v3=all(original['conditions'][c]==config['conditions'][c] for c in saved_e1),
        existing_parameter_files_unchanged=all(hashes[p]==digest for p,digest in initial.items() if p.startswith('params/')),
        global_costmap_unchanged=render_parameters(params,original,'E2_environment')['global_costmap']==render_parameters(params,config,'E2_environment')['global_costmap'],
        baseline_controllers_equal_v3_selected_except_samples=previous_controllers==config['conditions']['E2_environment']['controller_overrides'],
        tooling_tests_passed=int(re.search(r'(\d+) passed',test_log)[1]),
        physical_trials=0)
    save(o/'validation.json',validation)
    if not all(validation[k] for k in ('E1_rendered_equal_to_initial','E1_config_equal_to_v3',
                                      'existing_parameter_files_unchanged','global_costmap_unchanged',
                                      'baseline_controllers_equal_v3_selected_except_samples')):
        raise RuntimeError('Preservation validation failed')
    reproduce='''#!/usr/bin/env bash
set -euo pipefail
package=${DWVP_E2_PACKAGE:-/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation}
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dwvp-e2-v4-mpl
python3 "$package/scripts/dwvp_access_e2_v4_run.py" \\
  --work "${DWVP_E2_WORK:-/tmp/dwvp-e2-design-v4-reproduction}" \\
  --output "${DWVP_E2_REPORT:-/tmp/dwvp-e2-design-v4-report-reproduction}"
'''
    (o/'reproduce.sh').write_text(reproduce)
    resolution_lines=['| 条件 | 分解能 [m] | 完走 | 直線巡航 [m/s] | 位置平均/最大 [m] | 方位平均/最大 [deg] | plugin 計算 [ms] |',
                      '|---|---:|---:|---:|---|---|---:|']
    for label,r in resolution.items():
        g=r['group']; fmt=lambda k,d:base.format_stat(g[k],d)
        resolution_lines.append(f"| {label} | {dict(A=.05,B=.01,C=.02)[label]} | {g['successes']}/3 | {fmt('straight_cruise_mean_speed_m_s',3)} | {fmt('eval_mean_position_error_m',4)} / {fmt('eval_max_position_error_m',4)} | {fmt('eval_mean_heading_error_deg',2)} / {fmt('eval_max_heading_error_deg',2)} | {fmt('compute_time_mean_ms',3)} |")
    resolution_md='\n'.join(resolution_lines)
    (o/'resolution_table.md').write_text(resolution_md+'\n')
    stop_lines=[]
    for label,r in resolution.items():
        for stop in r['stops']:
            x,y=stop['position']; stop_lines.append(f"- {label} / {stop['trial']}: ({x:.4f}, {y:.4f}) m、観測 {stop['duration_s']:.2f} s、ゴール距離 {stop['goal_distance_m']:.3f} m。")
    load_lines=['| 条件 | セル数 | costmap更新遅延警告 | control周期遅延警告 | reset時等のsensor範囲外警告 | 受信間隔 平均/p95/最大 [s] |',
                '|---|---|---:|---:|---:|---|']
    for label,load in loads.items():
        p=load['publication_interval_s']
        load_lines.append(f"| {label} | {load['cell_dimensions']} | {load['update_loop_missed_warnings']} | {load['control_loop_missed_warnings']} | {load['sensor_origin_outside_warnings']} | {p['mean']:.3f}/{p['p95']:.3f}/{p['maximum']:.3f} |")
    compute_lines=['| 検証 | 手法 | plugin全呼び出し 平均/p95/最大 [ms] | 33.333 ms超過 |',
                   '|---|---|---|---:|']
    for label,load in loads.items():
        for method,t in load['plugin_calls'].items():
            compute_lines.append(f"| {label} | {method} | {t['mean']:.3f}/{t['p95']:.3f}/{t['maximum']:.3f} | {t['over_33p333_ms']}/{t['n']} |")
    r=facts['checks']; m=facts['metadata']; v=comparison['v4']; prior=comparison['v3']
    gap=r['facing_corner_gaps'][0]; counts=r['excess_interval_counts']; decision=base.read_json(w/'geometry_decision.json')
    selection=sweep['selected']['DWB']
    selected_text=(f"`{selection['candidate']}`、得点 {selection['score']:.4f}。" if selection else
                   '完走候補がなく選定なし。事前規則により v3 の 3 s / 16 / 64 を最終比較で維持した。')
    a,b=resolution['A']['group'],resolution['B']['group']
    outcome=f"A は {a['successes']}/3、B は {b['successes']}/3 完走。"
    if b['successes']==3 and a['successes']<3:
        outcome+='この経路と固定設定では分解能変更で完走を回復した。格子の量子化を原因候補とする仮説と整合するが、単独の原因を証明する比較ではない。'
    elif b['successes']<3:
        outcome+='0.01 m への変更だけでは安定完走に至らなかった。'
    diag_text='B が全試行完走したため、条件付きの 0.02 m 試験と停止時診断は実施していない。'
    if diagnosis is not None:
        ranges=[x['spread'] for x in diagnosis['goal_dist_ranges'] if x]
        diag_text=(f"B の別試行で5秒間購読し、{diagnosis['captured_evaluations']}件を保存した。"
                   f"GoalDist候補間の得点差の範囲は {min(ranges):.6g}–{max(ranges):.6g}。" if ranges else
                   '停止時購読の有効なGoalDist評価記録は得られなかった。')
        diag_text+=' `dwb_diagnosis.json`と元の評価JSONLに候補数、棄却critic、位置、最良速度を保存した。診断試行は選定と計算時間集計から除外した。'
        straight=diagnosis['first_straight_candidates']
        stationary=next((x for x in straight if abs(x['vx'])<1e-9),None)
        moving=max(straight,key=lambda x:x['vx']) if straight else None
        if stationary and moving and moving['vx']>0:
            z,p=stationary['scores'],moving['scores']
            diag_text+=(f"\n\n記録先頭の直進候補では、停止のGoalDist/GoalAlignが {z['GoalDist']:g}/{z['GoalAlign']:g}、"
                        f"vx={moving['vx']:.6f} m/sの候補は {p['GoalDist']:g}/{p['GoalAlign']:g} だった。"
                        f"合計得点は停止 {stationary['total']:.6f}、移動 {moving['total']:.6f}。")
            scales=diagnosis['first_best_scores']
            grid_delta=sum((p[k]-z[k])*scales[k]['scale'] for k in ('GoalDist','GoalAlign'))
            if abs(grid_delta)<1e-6 and p['GoalDist']!=z['GoalDist']:
                diag_text+=('この直進候補ではGoalDistに差があるがGoalAlignと相殺している。'
                            '「GoalDistが全候補で同点」だけではこの停止を説明できない。')
            if all(not r for r in diagnosis['rejections']) and all(np.linalg.norm(v[:2])<1e-9 for v in diagnosis['best_velocities']):
                diag_text+='この評価窓では棄却なし、最良並進速度は0だった。'
            diag_text+='別の停止状態への一般化は未検証。'
    readme=f'''# E2 route graph v4: 分解能比較と4手法の合成実験

作成日 2026-10-14。実際のHumble controller plugin、静的地図、理想ray scan、運動学的fake robotを用いた合成実験。物理試行は0。{outcome}

## 変更・新設ファイル

変更: {', '.join('`'+p+'`' for p in changes['changed'])}。
新設: {', '.join('`'+p+'`' for p in changes['new'])}。
実装の写しは`implementation/`、ハッシュは`source_manifest.json`。DWB criticのコード、生成器、1/30 sの窓、critic構成と比較A/Bのscaleはv3から維持した。MPPIはAngle 6 / Align 30 / wz_std 0.2で固定し再掃引していない。

## 経路とcheck-path

経由点は (2.5,0.05) → (−0.05,0.05) → (−0.05,{decision['lower_y_m']}) → (2.5,{decision['lower_y_m']}) m。NavFnは生成・実行していない。下側角から0.5 m以内のclearanceは {decision['clearance_near_requested_lower_corner_m']:.4f} m で、y=−2.6 mを採用した。正対はy∈[−0.8,−1.8]、yaw=π、遷移長0.35 m。

- {r['points']}点、長さ {r['length_m']:.4f} m、最大標本間隔 {r['spacing_m']['max']:.6f} m。直線の接線yawは厳密に一定、円弧半径は解析値0.3 m。
- R_min={r['xy_min_radius_m']:.6f} m（離散弦からの推定）。開始yaw=πとCSVの差は {m['start_yaw_minus_csv_yaw_rad']:.3g} rad。
- 0.22 m/s・上限0.6 rad/sで、接線の超過 {r['tangent_yaw']['excess_count']}点、正対込み {r['excess_count']}点。後者は円弧 {counts['corner']}、正対遷移 {counts['facing_transition']}、区間外 {counts['outside_expected_intervals']}点。最大要求角速度は {r['max_abs_required_yaw_rate_rad_s']:.4f} rad/s。
- 遷移と円弧の弧長の隙間: 入口 {gap['entry_gap_m']:.6f} m、出口 {gap['exit_gap_m']:.6f} m。両方0.1 m以上、正対端の追加変更なし。
- 最小clearance {r['clearance']['min_m']:.6f} m、位置 ({r['clearance']['position'][0]:.6f}, {r['clearance']['position'][1]:.6f}) m。棚側 {v['shelf']['min_m']:.6f} m、机側 {v['desk']['min_m']:.6f} m。
- v3(b)の同じ空間マスクでの最小値: 棚側 {prior['shelf']['min_m']:.6f} m、机側 {prior['desk']['min_m']:.6f} m。棚・机はy∈[−2.6,0.05]内の占有セルをx=−0.05で分割した呼称。地図は机の脚を含むが天板の張り出しは表していない。距離は中心から占有セル矩形までで、外周余裕は0.22 mを引いた値。
- ネイティブNav2 inflation検査は0.05 m/0.01 mとも最大cost=0、致死/内接/未知・範囲外=0。両CSVはバイト単位で同一。詳細は`B_graph/check/`と`inspection_005/check/`。

## 分解能の適用と計算負荷

`dwvp_access_experiment.render_parameters`はE2の`local_costmap_resolution_m`をlocal costmapだけに展開する。fake robot側の実行時snapshotでも値を照合した。local窓は3×3 m、robot_radius=0.22 m、inflation=0.3 m、cost_scaling_factor=3.0、更新5 Hz・配信2 Hzの設定を維持。global costmap設定はv3と同一で分解能0.05 m。

経路検査の`inspection_costmap_resolution_m`は実験用global costmapを変更しない。検査専用の静止rolling窓を地図全域に置き、元の0.05 m地図をネイティブStaticLayerで0.01 mへ標本化してInflationLayerの結果を購読する。元地図ファイルは変更していない。

{chr(10).join(load_lines)}

costmapの内部更新周期は直接計測していない。上表の周期は試行中に観測した購読callbackの受信間隔で、5 Hz更新の証明には使えない。Humbleの`metadata.update_time`は0だったため更新時刻・更新遅延量は取得できず、更新周期の遅れはnativeの`Map update loop missed`警告数を併記した。受信間隔が長い原因をcostmap更新、配信、購読処理のいずれかには特定していない。sensor範囲外警告はfake robotの反復開始時teleportに伴う窓位置の過渡を含む。全警告と記録は`costmap_and_compute.json`に保存した。

{chr(10).join(compute_lines)}

計算時間は`TimedController`が内側pluginの`computeVelocityCommands`全体を囲んだ計測であり、DWBの`prepare`に含まれるGoalDist/GoalAlignの格子伝播と候補採点を含む。costmap自身の別スレッド更新処理は含まない。表のp95/最大は全呼び出し、以下の±SDは各試行平均を3反復で集計した値。

## DWBの分解能比較

全条件はsim_time=3 s、PathOrientationCritic.scale=16、PathPositionCritic.scale=64、標本7/7/7。各3反復。

{resolution_md}

未完走の位置・方位誤差と巡航速度は停止までの部分軌道の値であり、完走時の精度と比較できない。停止位置は次のとおり。

{chr(10).join(stop_lines) if stop_lines else '未完走なし。'}

{diag_text}

## DWBの8条件掃引と選定

分解能0.01 m、標本7/7/7、各1試行。試行前に`plan_before_results.json`へ全候補と規則を記録した。完走候補の位置・方位誤差の時間積分をそれぞれ候補内最小値で正規化し、和が最小の条件を選ぶ。同点は事前の列挙順。計算負荷の干渉を避けるため全試行を逐次実行した。

{sweep_md}

選定: {selected_text} 最終設定は`selected_config.yaml`。

## 4手法各3反復

全手法でlocal costmapは0.01 m。EC-DWVPはτ_y=0.6 s、L=0.11 m固定。±は標本SD（ddof=1）。位置・方位誤差は共通評価区間s∈[0,経路長−0.6 m]、走行時間はFollowPath完走まで。直線巡航はs∈[0.3,経路長−0.6 m]かつ参照接線の微分の絶対値≤0.1 rad/mで抽出したodometry速度ノルムの平均。正対遷移が直線上にある場合もこの定義には含まれる。

{final_table}

各反復の成功、評価区間完了、data_errors、plugin失敗数は`B_graph/verify_report.json`と`statistics.json`を参照。未完走行の有限値は部分観測であり、走行時間は欠測として示す。

## 図・生データ・再実行・検証

- `B_graph/map_tracking.png`: 地図、経路、正対と遷移、4手法の全3反復。
- `B_graph/heading_speed.png`: 弧長に対する方位誤差と並進速度。
- `B_graph/check/path_check.png`: 方位、clearance、曲率、要求角速度。0.05 m検査図は`inspection_005/check/path_check.png`。
- `B_graph/route/`: 経路CSV、正対前の厳密な接線CSV、metadata、元地図の写し、設定、inflation costmap。
- `B_graph/sweep_raw.tar.gz`と`verify_raw.tar.gz`: 全CSV、設定snapshot、判定、timing、ログ。分解能比較は`resolution_A/`、`resolution_B/`（必要時`resolution_C/`）に非圧縮で保存。元の作業先は`{w}`。
- `bash reproduce.sh`: 未使用の/tmpへbuildから図表まで再実行。既存Docker imageとローカル依存だけを使用し、Dockerは`e2_tooling.sh --network none`経由。非完走も結果として保存する。ネットワーク、サブエージェント、物理ロボット、commit、push、branch切替、作業ツリー削除は使用していない。
- 単体テストに遷移と円弧の隙間検査を追加。`e2_tooling.sh test`: {validation['tooling_tests_passed']} passed。E1設定と作業前の全E1展開結果、既存YAML、global costmapの不変を確認した（`validation.json`）。

## 未確定事項

格子量子化仮説の因果関係、実機での通行可否、測位誤差、動力学、人への不快感はこの合成実験だけでは確定しない。costmap内部更新の直接の周期計測も未実施。地図は0.05 mのままで、local costmapの0.01 m化は元地図の幾何精度を向上させない。MPPIの乱数と実時間スケジューリングにより結果は変動する。各1試行の選定と3反復から統計的優越性は判断しない。DWBは並進ノルム0.22 m/s、他手法の共通制限は各軸±0.22 m/sであり、斜行速度の上限は同一ではない。
'''
    (o/'README.md').write_text(readme)
    save(o/'SHA256SUMS.json',{str(p.relative_to(o)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in o.rglob('*') if p.is_file()})
    print(final_table)
    print(resolution_md)


if __name__=='__main__':
    main()
