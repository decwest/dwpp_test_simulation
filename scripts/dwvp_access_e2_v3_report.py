#!/usr/bin/env python3
"""Export both v3 routes, recorded DWB diagnosis, frozen sweep and repeat results."""
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

ROOT = Path(__file__).resolve().parents[1]
CHANGED = ['scripts/dwvp_access_path.py', 'scripts/dwvp_access_e2.py',
           'scripts/dwvp_access_experiment.py', 'scripts/dwvp_access_e2_report.py',
           'scripts/e2_tooling.sh', 'tests/test_access_path.py', 'tests/test_access_e2.py']
NEW = ['scripts/dwvp_access_e2_v3_report.py', 'params/dwvp_access_path_v3.yaml',
       'params/dwvp_access_e2_route_design_v3.yaml']


def diagnosis(work):
    results = {}
    for name in ('diagnostic_v2', 'diagnostic_v2_vy20', 'baseline_original', 'diagnostic_original'):
        report = work/name/'report.json'
        if not report.exists():
            continue
        row = base.read_json(report)['rows'][0]
        folder = base.host_path(row['session'], work)/'runs'/row['trial_id']
        track = base.read_csv(folder/'tracking.csv')
        path = np.loadtxt(base.host_path(row['session'], work)/'paths/E2_environment.csv', delimiter=',', skiprows=1)
        valid = np.isfinite(np.c_[track['x'], track['y'], track['yaw']]).all(axis=1)
        track = track[valid]
        positions = np.c_[track['x'], track['y']]
        distance = np.linalg.norm(positions-path[-1, :2], axis=1)
        yaw_error = np.abs(np.arctan2(np.sin(track['yaw']-path[-1, 2]), np.cos(track['yaw']-path[-1, 2])))
        evaluations = work/name/'controllers/dwb_evaluation.jsonl'
        scores = [json.loads(line) for line in evaluations.read_text().splitlines()] if evaluations.exists() else []
        rejected = {}
        for evaluation in scores:
            for critic, count in evaluation['rejected_by_critic'].items():
                rejected[critic] = rejected.get(critic, 0)+count
        log = (work/name/'controllers/stack/controller_server.log').read_text()
        results[name] = dict(success=row['success'], row=row, stop_position=positions[-1].tolist(),
            stop_yaw_rad=float(track['yaw'][-1]), goal_distance_m=float(distance[-1]),
            closest_goal_distance_m=float(distance.min()), goal_yaw_error_rad=float(yaw_error[-1]),
            entered_xy_goal_window=bool(np.any(distance <= .1)),
            final_simple_goal_checker_xy=bool(distance[-1] <= .1),
            final_simple_goal_checker_yaw=bool(yaw_error[-1] <= .3),
            state_source='Reconstructed from recorded pose and runtime SimpleGoalChecker stateful=false, xy=.1, yaw=.3',
            progress_failure='Failed to make progress' in log,
            rejection_counts_in_capture=rejected, captured_evaluations=len(scores),
            rotate_to_goal_nonzero_best_scores=sum(e['best_scores'].get('RotateToGoal', {}).get('raw', 0) != 0 for e in scores),
            zero_rejection_capture_samples=sum(not e['rejected_by_critic'] for e in scores))
    return results


def check_table(facts):
    lines = ['| 経路 | 長さ [m] | R_min [m] | 角速度超過 接線/正対込み [点] | 最小 clearance [m] | 最大 inflation cost | 致死/内接/未知セル |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for f in facts:
        r = f['checks']; c = r['inflation']
        lines.append(f"| {f['route']} | {r['length_m']:.4f} | {r['xy_min_radius_m']:.5f} | {r['tangent_yaw']['excess_count']}/{r['excess_count']} | {r['clearance']['min_m']:.5f} | {c['max_published_cost']} | {c['lethal_count']}/{c['inscribed_count']}/{c['unknown_or_outside_count']} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); w, o = args.work, args.output
    o.mkdir(parents=True, exist_ok=False)
    base.METHODS = ['VP_CLIP', 'DWVP', 'MPPI', 'DWB']
    base.LABELS['DWVP'] = 'EC-DWVP'
    base.METRICS.append('straight_cruise_mean_speed_m_s')
    facts = []
    for name, route, check, verify in [('B_graph', 'B_graph_requested', 'B_check_final', 'B_verify_final'),
                                        ('A_navfn', 'A_navfn', 'A_check', 'A_verify')]:
        f = base.export_route(name, w/route, w/check, w/'B_sweep', w/verify, w, o)
        (o/name/'results_table.md').write_text(result_table(f['groups'])+'\n')
        facts.append(f)
    sweep = base.read_json(w/'B_sweep/report.json')
    diag = diagnosis(w)
    (o/'dwb_diagnosis.json').write_text(json.dumps(diag, indent=2, allow_nan=False)+'\n')
    for name in ('diagnostic_v2.yaml', 'diagnostic_v2_vy20.yaml', 'diagnostic_original.yaml',
                 'selected_config.yaml', 'reproduce.sh', 'build-final.log', 'critic-test-final.log',
                 'tooling-test-final.log', 'runtime-provenance.txt', 'initial_parameter_hashes.json', 'historical_diagnostics.txt'):
        if (w/name).exists():
            shutil.copy2(w/name, o/name)
    shutil.copytree(w/'diagnostic_v2_route', o/'diagnostic_v2_route')
    shutil.copytree(w/'nav2_source_evidence', o/'nav2_source_evidence')
    for name in ('sweep.csv', 'plan_before_results.json'):
        shutil.copy2(w/'B_sweep'/name, o/name)
    with tarfile.open(o/'diagnosis_and_development_raw.tar.gz', 'w:gz') as archive:
        for name in (*diag, 'B_graph', 'B_graph_initial', 'B_sweep', 'B_verify',
                     'diagnostic_reset', 'diagnostic_goalalign', 'diagnostic_no_goalalign',
                     'diagnostic_initial_grid', 'diagnostic_oscillation', 'diagnostic_oscillation_repeat'):
            if (w/name).exists(): archive.add(w/name, arcname=name)
        for file in w.glob('*.log'):
            archive.add(file, arcname=file.name)
    for rel in CHANGED+NEW:
        dest = o/'implementation'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    initial = base.read_json(w/'initial_parameter_hashes.json') if (w/'initial_parameter_hashes.json').exists() else {}
    original = yaml.safe_load((ROOT/'params/dwvp_access_experiment.yaml').read_text())
    config = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v3.yaml').read_text())
    e1_equal = all(render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', original, c) ==
                   render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', config, c)
                   for c in original['conditions'] if c.startswith('E1'))
    tool_log = (w/'tooling-test-final.log').read_text(); critic_log = (w/'critic-test-final.log').read_text()
    validation = dict(E1_rendered_parameters_equal=e1_equal,
        preexisting_parameter_files_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in initial.items()),
        tooling_tests_passed=int(re.search(r'(\d+) passed', tool_log)[1]),
        critic_gtests_passed=int(re.search(r'\[  PASSED  \] (\d+) tests', critic_log)[1]),
        physical_trials=0, route_radius_target_met=.25 <= facts[0]['checks']['xy_min_radius_m'] <= .35)
    (o/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    files = [p for folder in ('scripts', 'params', 'tests', 'src') for p in (ROOT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [p for p in (ROOT.parent/'dwb_path_tracking_critics').rglob('*') if p.is_file()]
    (o/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)+'\n')
    diagnostic_names = {
        'B_verify': '初回4手法反復・既定値',
        'diagnostic_reset': 'costmap消去',
        'diagnostic_goalalign': 'GoalAlign.forward_point_distance=0.525',
        'diagnostic_no_goalalign': 'GoalAlign.scale=0',
        'diagnostic_oscillation': 'Oscillation.reset_dist=0.005・単独',
        'diagnostic_oscillation_repeat': 'Oscillation.reset_dist=0.005・反復',
    }
    additional = ['| 追加診断 | DWB完走 |', '|---|---:|']
    additional_facts = {}
    for name, label in diagnostic_names.items():
        report = w/name/'report.json'
        if report.exists():
            rows = [r for r in base.read_json(report)['rows'] if r['controller'] == 'DWB']
            additional_facts[name] = dict(label=label, successes=sum(r['success'] for r in rows), trials=len(rows))
            additional.append(f"| {label} | {sum(r['success'] for r in rows)}/{len(rows)} |")
    (o/'additional_diagnosis.json').write_text(json.dumps(additional_facts,indent=2)+'\n')
    b, a = facts
    failure = diag['diagnostic_v2']; fixed = diag['diagnostic_v2_vy20']
    counts = b['checks']['excess_interval_counts']
    diag_lines = ['| 条件 | 完走 | 停止位置 [m] | ゴール距離 [m] | goal XY 窓に入ったか |', '|---|---:|---|---:|---:|']
    for key, label in [('diagnostic_v2','v2経路・vy=5・free map'), ('diagnostic_v2_vy20','v2経路・vy=20・free map'), ('baseline_original','円弧経路・vy=20・実験室地図')]:
        r=diag[key];diag_lines.append(f"| {label} | {r['success']} | ({r['stop_position'][0]:.4f}, {r['stop_position'][1]:.4f}) | {r['goal_distance_m']:.4f} | {r['entered_xy_goal_window']} |")
    outcome = '、'.join(f"{f['route']}: "+', '.join(f"{base.LABELS[g['method']]} {g['successes']}/{g['repeats']}" for g in f['groups']) for f in facts)
    lower_arc = b['metadata']['graph']['corners'][-1]
    facing_interval = b['metadata']['facing']['intervals'][-1]
    lower_overlap = max(0., facing_interval['transition_end_m']-lower_arc['start_m'])
    geometric_notes = []
    for f in facts:
        m=f['metadata'];r=f['checks'];c=r['clearance']
        path=np.loadtxt(w/('B_graph_requested' if f['route']=='B_graph' else 'A_navfn')/'E2_environment.csv',delimiter=',',skiprows=1)
        geometric_notes.append(f"- {f['route']}: 最小 clearance の位置 ({c['position'][0]:.5f}, {c['position'][1]:.5f}) m、弧長 {c['progress_m']:.5f} m。開始 yaw={m['start'][2]:.8f} rad、CSV との差={m['start_yaw_minus_csv_yaw_rad']:.8f} rad。終端接線 yaw={path[-1,2]:.8f} rad。最大要求角速度={r['max_abs_required_yaw_rate_rad_s']:.4f} rad/s。CSV始点=({path[0,0]:.5f},{path[0,1]:.5f}) m。")
    timings=[]
    for f in facts:
        rows=base.read_json(w/('B_verify_final' if f['route']=='B_graph' else 'A_verify')/'report.json')['rows']
        for method in base.METHODS:
            samples=[r for r in rows if r['controller']==method]
            timings.append(f"- {f['route']} / {base.LABELS[method]}: 評価区間到達 {sum(bool(r.get('evaluation_complete')) for r in samples)}/3、data_errors あり {sum(bool(r.get('data_errors')) for r in samples)}/3、失敗 plugin 呼び出し {sum(r.get('timing',{}).get('failed_calls',0) for r in samples)} 回。")
    text=f'''# 実験2 経路v3と4手法の合成実験

円弧経路 (b) は要求 waypoint を変更せず、半径0.3 mの円弧2本で生成できた。最小半径は {b['checks']['xy_min_radius_m']:.5f} m で目標を満たす。比較案 (a) も指定の通路内1点viaで生成・検査できた。合成実験の完走数は {outcome}。**DWBの未完走を最小設定変更だけで安定して解消する課題は未達**であり、成功した単独試行と失敗した反復を併記する。

本結果は実 Humble plugin、共通 velocity smoother、合成運動学モデルによる実験であり、物理HSRの測定ではない。v2の合成環境は障害物のない地図を配信していた。v3では inflation の挙動を評価するため、指定の実験室地図と、その地図に対する理想的な平面ray scanを全手法に配信した。この変更を含むため、v2とv3の差をinflation半径だけの効果として扱わない。外部ネットワークと物理ロボットへの接続は行っていない。

## 経路生成と検査

(b) は (2.5,0.05) → (−0.15,0.05) → (−0.15,−2.5) → (2.5,−2.6) を折れ線とし、両隣の直線に接する半径0.3 mの円弧を解析的に挿入した。NavFnとsmootherを起動せず、独立したNav2 costmap nodeを衝突検査にだけ使う。内角が鋭く接点が隣接辺を使い切る場合は半径を比例縮小する。今回の2角では縮小なし。完全な180°反転は有限半径の接円弧を定義できないため入力エラーにする。解析弧長を最大0.02 mで等間隔に標本化し、直線と円の解析接線をyawに使う。円弧の中心・半径・接点・弧長区間は `B_graph/route/path_metadata.json`。

(a) は start (2.5,0.05,π) → via (0,−1.5) → goal (2.5,−2.6,0) の2区間をNavFnで生成し、連結後にSimpleSmootherの既定重み w_data=0.2、w_smooth=0.3 を適用した。0.02 mで再標本化し、従来どおり前後0.10 mの差分で接線yawを推定した。`--align-start-yaw` で開始yawをCSVの初期接線へ合わせた。終端も生成接線を使うため、指定goal yaw=0は保持しない。NavFnが始点をセルに丸める既存の挙動も残り、CSV始点(2.49000,0.07000) mに対し、機体の開始位置は指定どおり(2.5,0.05) mとした。

両経路とも線分 (−0.15,−0.8)–(−0.15,−1.9) の正対yaw=π、遷移長0.35 mを適用した。方位遷移は最短角差のsmoothstep補間。地図の占有・未知・地図外セルに対し、標本間も細分した半径0.22 mの円形footprint検査を通過した。生成前の折れ線上のviaは円弧へ置き換えられるため、丸め前のviaセルのコストと最終経路の通行判定は異なる。waypointの移動は不要だった。

{check_table(facts)}

角速度検査は並進速度0.22 m/s、上限±0.6 rad/s。半径0.3 mの円弧自体が約0.733 rad/sを要求するため、角で減速が必要となる。正対の90°smoothstep遷移も最大約1.48 rad/sを要求する。半径目標の達成は一定速度での角速度制約充足を意味しない。(b) の超過71点の内訳は角 {counts['corner']} 点、正対遷移 {counts['facing_transition']} 点、その他 {counts['outside_expected_intervals']} 点。接線yawのみの超過は {b['checks']['tangent_yaw']['excess_count']} 点で角に限られた。差分の境界判定には標本間隔、接線推定には前後0.10 mの窓幅を許容し、重複した点は角へ分類した。指定の正対位置と遷移長を維持すると、下側の方位遷移末尾と円弧冒頭が弧長で約 {lower_overlap:.3f} m重なる。両者を完全に分離する配置調整は行っていない。

{chr(10).join(geometric_notes)}

clearanceは地図の占有セル矩形から機体中心までの距離であり、机・椅子の脚、棚、壁を含む。著者確認では天板の張り出しはなく、椅子を地図取得時の位置に固定する。(b) の円形外周余裕は最小約 {(b['checks']['clearance']['min_m']-.22)*1000:.1f} mmにとどまる。測位誤差や実機の配置差を含む通行余裕ではない。

## E2のinflation設定

`params/dwvp_access_path_v3.yaml` の planner.inflation_radius_m と、`params/dwvp_access_e2_route_design_v3.yaml` の E2_environment.costmap_inflation_radius_m を0.3 mに設定した。後者はE2の展開済みlocal/global costmapにだけ適用する。robot_radius=0.22 m、cost_scaling_factor=3.0は維持した。既存のfootprint_padding=0.01 mも維持しており、円形footprint検査とは扱う外周が厳密には同一ではない。E1の展開設定は元設定と同一で、既存の設定ファイルは変更していない。

inflationの表の値はNav2が公開するOccupancyGridの値（100=致死、99=内接領域、1〜98=非致死コスト）。内部の0〜254のraw costとは区別する。(b) は最大91、(a) は最大0。正対区間内の標本コストは両経路ともすべて0だった。全標本の値・cell indexは各 `route/inflation_check.json`、costmap全体は `inflated_costmap.csv`。共通のlocal costmapについても実行時パラメータを取得し、半径0.3 mとrobot_radius=0.22 mを照合した。

## DWB未完走の診断と最小変更

旧v2経路、方位重み16、位置重み64、sim_time=3 s、標本数20/5/20をfree-map条件で再現した。停止位置は ({failure['stop_position'][0]:.4f}, {failure['stop_position'][1]:.4f}) m、ゴールまで {failure['goal_distance_m']:.3f} mで、最初の角の手前にとどまった。サーバの終了理由は `Failed to make progress`。実行時SimpleGoalCheckerはstateful=false、XY許容0.1 m、yaw許容0.3 radで、XYのgoal窓に入っていない。RotateToGoalのbest scoreは取得した全標本で0だった。ゴール付近の減速・回転段階による停止ではない。

停止後の短い評価記録ではOscillationによる候補棄却が起きた一方、棄却ゼロの周期でも極低速候補を選び続けた。したがってOscillationの禁止だけでは停止を説明できない。格子型GoalDist/GoalAlignの得点が一定の領域で、連続位置・方位criticが近くの低速候補を選ぶ停滞を観測した。新criticの誤差が小さくても、先へ進めているとは限らない。評価記録と停止判定は `dwb_diagnosis.json`、原データは `diagnosis_and_development_raw.tar.gz` に保存した。

vy_samplesを5から20へ増やすと、旧経路・free map・同じ重みの単独試行と円弧経路の単独試行は完走した。しかし初回の4手法反復ではDWBが0/3完走にとどまり、標本数変更だけでは不十分だった。

costmapを消去しても開始直後の停止が残り、最初の50経路標本のraw costは消去前後とも0だった。GoalAlignの距離を0.525 mへ変更した3反復、GoalAlign.scale=0の単独試行も未完走だった。Oscillation.oscillation_reset_distを0.05 mから0.005 mへ変更した単独試行は完走したが、追試の3反復は1/3完走だった。この変更も安定した修正としては採用しない。初回反復と追加診断の原データは保存し、最終比較の集計には混ぜていない。

{chr(10).join(additional)}

`diagnostic_initial_grid` のnative評価記録では、直進速度の異なる複数候補がGoalDist/GoalAlignで同じ格子得点を持つことを確認した。一方、Oscillationによる棄却がゼロの周期にも低速候補を選んでいる。したがって、goal checkerやRotateToGoalの終了判定が原因とはいえず、格子得点が同じになる領域と状態付きの候補棄却が併存する低速停滞である。どの要因がどの失敗を支配したかを一意に特定し、設定だけで解消するところまでは到達していない。連続値の進捗criticの追加は比較構成を変えるため行わなかった。

最終DWB設定は依頼どおり標本数20/20/20とし、RotateToGoal、goal checker、GoalAlign、Oscillationの既存設定を維持した。選定は最初に固定した24条件の掃引と規則のままで、追加診断の成功例による再選定はしない。3反復で安定するという前提は置かず、未完走を結果として報告する。

{chr(10).join(diag_lines)}

DWBのnative evaluationは1周期に数千の軌跡を含む。常時購読した初回診断は実行を大きく遅延させ、円弧手前で失敗したため性能比較から除外した。最終診断は開始36秒後の5秒間だけ購読し、記録による影響を限定した。掃引・最終3反復ではevaluationを購読しない。取得したstateはposeとruntime checker設定から再構成したものであり、非公開の内部変数を直接読み出した記録ではない。原ソースはローカルにあったNav2 tag 1.1.19から取り出し、`nav2_source_evidence/` に保存した。

## 事前固定した掃引と選定

(b) だけでDWB 12条件（sim_time=2/3 s × yaw scale=4/16/64 × position scale=16/64）、MPPI 12条件（PathAngle=6/12/20 × PathAlign=15/30 × wz_std=0.2/0.4）を各1試行した。最初の掃引試行より前に `plan_before_results.json` を書き、完走候補の位置・方位誤差時間積分をそれぞれの最小値で正規化した和を最小化した。同点は格子順。失敗候補の途中までの小さい誤差は選定しない。共通評価窓は弧長0から末尾0.6 m手前。4つの独立ROS domainで掃引した。最終表のverifyは両経路・全試行を順次実行し、診断処理を並行させていない。初回(b)反復では途中から診断処理が並行したため、性能表には再実行したB_verify_finalだけを用いた。

### DWB

{sweep_table(sweep['rows'], 'DWB')}

### MPPI

{sweep_table(sweep['rows'], 'MPPI')}

### 選定値

```json
{json.dumps(sweep['selected'], indent=2, ensure_ascii=False)}
```

`selected_config.yaml` を両経路に共通で使用した。(a) では再掃引していない。DWBはLimitedAccelGenerator、sim_period=1/30 s、標本数20/20/20、max_speed_xy=0.22 m/s。native generatorはゼロ標本を追加することがあるため評価候補の総数は8000を超えることがある。MPPIは60 steps × 1/30 s=2 s。VP_CLIPは0.75 sの適応前方注視0.11〜0.33 m、EC-DWVPはτ_y=0.6 s、固定L=0.11 m。PP系のcost regulationとrotate-to-headingはオフ。

## 4手法の各3反復

平均±SDは各反復の指標を算術平均し、標本SD（ddof=1）を併記した。位置・方位の平均は評価窓内の時間積分を滞在時間で割り、最大も試行ごとに計算してから集計する。走行時間は停止確認まで。計算時間はTimedControllerによるplugin呼び出し時間で、サーバ全体の周期ではない。巡航速度は弧長[0.3, length−0.6] mのうちXY接線曲率の絶対値≤0.1 rad/mの直線部分に対応するodometry標本の平均で、直線上の正対方位遷移も含む。0.22未満は丸め前の値を比較した。

### (b) 円弧経路

{result_table(b['groups'])}

### (a) NavFn＋1 via

{result_table(a['groups'])}

{chr(10).join(timings)}

失敗試行がある行では、途中までに観測した有限値も保持している。その行を完走時の精度として比較できず、走行時間のnも他の指標と異なり得る。各試行の成功判定・評価窓完了・data_errors・失敗plugin呼び出し数は `verify_report.json` と `statistics.json` を参照する。DWBの20/20/20評価は計算量が増え、30 Hzの33.3 msを超える呼び出しと周期遅延を含む。この実時間の合成実験から、実機で30 Hzを維持できるとは主張しない。

## 保存物・検証・再実行

- `B_graph/map_tracking.png`、`A_navfn/map_tracking.png`: 地図・経路・正対区間と4手法の全3反復。
- `B_graph/heading_speed.png`、`A_navfn/heading_speed.png`: 弧長に対する方位誤差と並進速度。
- `B_graph/check/path_check.png`、`A_navfn/check/path_check.png`: 方位、clearance、曲率、要求角速度。
- 各 `route/`: CSV、metadata、地図の写し、inflation標本、生成設定とログ。
- 各 `sweep_raw.tar.gz`、`verify_raw.tar.gz`: パラメータsnapshot・全CSV・timing・判定ログ。(a) のsweep archiveは(b)と同じ選定データであり、追加掃引ではない。
- `reproduce.sh`: 新しい/tmp作業場所でbuild、経路生成・検査、基本診断、掃引、両経路verify、テスト、報告書出力を再実行する。追加の不採用設定の開発記録は元報告のarchiveから固定証跡としてコピーし、新規測定と区別する。
- `validation.json`: tooling {validation['tooling_tests_passed']} tests、critic {validation['critic_gtests_passed']} gtests、E1展開設定の同一性、既存YAMLの不変。
- `implementation/`、`source_manifest.json`、`SHA256SUMS.json`: 今回の実装の写しとハッシュ。元の生データは `{w}`。

この報告ディレクトリで `bash ./reproduce.sh` を実行する。既定の作業場所は `/tmp/dwvp-e2-design-v3-reproduction`、報告先は `/tmp/dwvp-e2-design-v3-report-reproduction`。どちらも未使用の場所を指定する。変更する場合は `DWVP_E2_WORK` と `DWVP_E2_REPORT` を設定する。既存Docker imageとローカル依存関係を使い、imageの取得は行わない。verifyに未完走があっても図表とテスト結果まで保存した後、非零終了する。乱数と実時間実行により再実行の数値や選定値は変わり得るため、診断の解釈は保存した元記録とも照合する。

変更ファイル: {', '.join('`'+p+'`' for p in CHANGED)}。新設ファイル: {', '.join('`'+p+'`' for p in NEW)}。既存criticの実装は変更していない。wrapperはクリーンな/tmpからもビルドできるよう、依存するcriticを先にビルドする順序へ修正した。Dockerの起動はすべて既存wrapperの`--network none`経由。commit・push・branch切替は行っていない。

## 未確定事項

DWBの安定完走は未達で、設定だけの修正を成功扱いにできない。(b) の最小外周余裕は約4 mmで、合成モデルの追従経路が参照から外れるとこの余裕を使い切り得る。各反復の実際の地図clearanceは `tracking_map_clearance.json` に示した。地図に理想センサを加えた合成完走は、実機の通行可否・測位誤差・人への不快感の評価ではない。共通制限は並進各軸±0.22 m/sで、DWBだけはさらに並進ノルム0.22 m/sの候補制限を持つ。MPPIの斜行ノルムが0.22 m/sを超える条件差はv2から維持している。MPPIの乱数と実時間スケジューリングによるばらつきがあり、各候補1試行の選定と3反復のSDから統計的優越性は判断しない。
'''
    (o/'README.md').write_text(text)
    (o/'all_facts.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')
    hashes={str(p.relative_to(o)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    (o/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(check_table(facts))
    for f in facts: print(f['route']+'\n'+result_table(f['groups']))


if __name__ == '__main__':
    main()
