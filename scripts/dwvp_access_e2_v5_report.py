#!/usr/bin/env python3
"""Summarize the v5 synthetic trials from recorded data, without ROS or network."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time

import numpy as np
import yaml

import dwvp_access_e2_report as base
from dwvp_access_e2_v2_report import result_table
from dwvp_access_e2_v5_run import ROOT, PAPER, V4, DEST, save, digest
from dwvp_access_experiment import render_parameters
from dwvp_access_metrics import project_tracking


def number(value, digits=3):
    return '—' if value is None or not math.isfinite(value) else f'{value:.{digits}f}'


def sweep_table(rows):
    header = ['GoalDist', '方位scale', 'sim_time [s]', '完走', '位置積分 [m s]', '方位積分 [deg s]',
              '選定得点', '巡航 [m/s]', 'plugin計算 [ms]']
    lines = ['| '+' | '.join(header)+' |', '| '+' | '.join(['---:']*len(header))+' |']
    for r in rows:
        o = r['overrides']
        cells = [f'{o[k]:g}' for k in ('GoalDist.scale', 'PathOrientationCritic.scale', 'sim_time')]
        cells += ['1/1' if r['success'] else '0/1']
        cells += [number(r.get(k), 4 if 'position' in k else 3) for k in
                  ('eval_position_error_integral_m_s', 'eval_heading_error_integral_deg_s', 'selection_score',
                   'straight_cruise_mean_speed_m_s', 'compute_time_mean_ms')]
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def profile(row, work, path):
    folder = base.host_path(row['session'], work)/'runs'/row['trial_id']
    tracking = base.read_csv(folder/'tracking.csv')
    valid = np.isfinite(np.c_[tracking['x'], tracking['y'], tracking['yaw'], tracking['stamp_s']]).all(axis=1)
    tracking = tracking[valid]
    projected = project_tracking(np.c_[tracking['x'], tracking['y'], tracking['yaw']], path)
    odom = base.read_csv(folder/'odom.csv')
    odom = odom[(odom['stamp_s'] >= tracking['stamp_s'][0]) & (odom['stamp_s'] <= tracking['stamp_s'][-1])]
    progress = np.interp(odom['stamp_s'], tracking['stamp_s'], projected[:, 2])
    return tracking, projected, progress, np.hypot(odom['vx'], odom['vy'])


def facing_stats(rows, work, path, meta):
    results = []
    for r in rows:
        _, projected, _, _ = profile(r, work, path)
        mask = np.zeros(len(projected), dtype=bool)
        for interval in meta['facing']['intervals']:
            mask |= (projected[:, 2] >= interval['start_m']) & (projected[:, 2] <= interval['end_m'])
        errors = np.abs(np.rad2deg(projected[mask, 1]))
        results.append(dict(controller=r['controller'], repeat=r['repeat'], success=r['success'],
            trial_id=r['trial_id'], samples=len(errors),
            full_facing_interval_traversed=bool(projected[:, 2].max() >= max(i['end_m'] for i in meta['facing']['intervals'])),
            mean_abs_heading_error_deg=float(errors.mean()) if len(errors) else None,
            max_abs_heading_error_deg=float(errors.max()) if len(errors) else None))
    return results


def sweep_figure(rows, work, path, meta, output):
    plt = base.plt
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True, sharey='row')
    for i, r in enumerate(rows):
        band, col = 2*(i//4), i % 4
        _, projection, progress, speed = profile(r, work, path)
        o = r['overrides']
        axes[band, col].plot(projection[:, 2], np.abs(np.rad2deg(projection[:, 1])), color='#0072B2', lw=1)
        axes[band+1, col].plot(progress, speed, color='#009E73', lw=1)
        axes[band, col].set_title(f"GoalDist {o['GoalDist.scale']:g}, yaw {o['PathOrientationCritic.scale']:g}, T={o['sim_time']:g}s\n"+
                                  ('Completed' if r['success'] else 'Incomplete (partial trajectory)'), fontsize=10)
        axes[band+1, col].axhline(.22, color='black', ls=':', lw=.8)
        for a in axes[band:band+2, col]:
            a.grid(alpha=.25); a.set_xlim(0, arc[-1])
            a.axvline(arc[-1]-.6, color='gray', ls='--', lw=.8)
            for interval in meta['facing']['intervals']:
                a.axvspan(interval['start_m'], interval['end_m'], color='#D55E00', alpha=.1)
                a.axvline(interval['transition_start_m'], color='#D55E00', ls=':', lw=.8)
                a.axvline(interval['transition_end_m'], color='#D55E00', ls=':', lw=.8)
        axes[band+1, col].set_xlabel('Projected arc length [m]')
    for row in (0, 2):
        axes[row, 0].set_ylabel('Absolute heading error [deg]')
        axes[row+1, 0].set_ylabel('Translation speed [m/s]')
    fig.suptitle('DWB v5 sweep | GoalAlign=0, position scale=64, resolution=0.01 m | one trial per condition\n'
                 'Shaded: fixed facing; orange dotted: transitions; gray dashed: evaluation end', fontsize=12)
    fig.tight_layout(); fig.savefig(output/'dwb_sweep_heading_speed.png', dpi=170); plt.close(fig)


def diagnose(work):
    evidence = []
    sweep = base.read_json(work/'sweep/report.json')['rows']
    path = np.loadtxt(work/'route/E2_environment.csv', delimiter=',', skiprows=1)
    for original in sorted(work.glob('diagnostic_goal_*')):
        if not original.is_dir() or not re.fullmatch(r'diagnostic_goal_(24|96)', original.name):
            continue
        retry = work/(original.name+'_retry')
        folder = retry if retry.is_dir() else original
        records = [json.loads(line) for line in (folder/'controllers/dwb_evaluation.jsonl').read_text().splitlines()]
        if not records:
            raise RuntimeError(str(folder)+': no diagnostic evaluations; repeat with an earlier capture window')
        rows = base.read_json(folder/'report.json')['rows']
        entry = dict(stage=folder.name, captured_cycles=len(records), capture_window_s=5.,
                     excluded_from_selection_and_timing=True, success=rows[0]['success'],
                     retried_empty_capture=folder != original,
                     initial_capture_source=str(original/'controllers/dwb_evaluation.jsonl'))
        goal = float(original.name.rsplit('_', 1)[1])
        failed = next(r for r in sweep if not r['success'] and r['overrides']['GoalDist.scale'] == goal)
        tracking, projected, _, _ = profile(failed, work, path)
        end = tracking[tracking['stamp_s'] >= tracking['stamp_s'][-1]-5.]
        displacement = float(np.linalg.norm([end['x'][-1]-end['x'][0], end['y'][-1]-end['y'][0]]))
        entry['sweep_failure'] = dict(candidate=failed['candidate'],
            status=failed.get('status'),
            final_pose=[float(tracking[k][-1]) for k in ('x', 'y', 'yaw')],
            final_projected_s_m=float(projected[-1, 2]), final_5s_displacement_m=displacement,
            duration_s=failed.get('duration_s'), evaluation_complete=failed.get('evaluation_complete'),
            final_position_error_m=failed.get('final_position_error_m'))
        if records:
            # Prefer an observed stationary cycle, then the last recorded cycle.
            stationary = [r for r in records if np.linalg.norm(r['best_velocity']) < .01]
            cycle = (stationary or records)[-1]
            forward = [r for r in cycle['straight_candidates'] if r['vx'] > 1e-6 and r['total'] >= 0]
            complete = [candidate for candidate in forward if 'PathPositionCritic' in candidate['scores']]
            chosen = min(complete or forward, key=lambda r: r['total']) if forward else None
            if chosen:
                chosen['velocity'] = [chosen['vx'], 0., 0.]
            forward_note = 'vx>0, vy=wz=0; prefer complete scores, then minimize total.'
            if chosen is None and cycle['best_velocity'][0] > 1e-6:
                # During a turn the acceleration window may contain no wz=0
                # candidate. The observed best trajectory itself moves forward.
                chosen = dict(velocity=cycle['best_velocity'], total=cycle['best_total'],
                              scores={k:v['raw'] for k,v in cycle['best_scores'].items()})
                forward_note = 'No vx>0, vy=wz=0 candidate was recorded; the best trajectory is itself a forward-moving turning candidate.'
            scales = {k: v['scale'] for k, v in cycle['best_scores'].items()}
            entry.update(cycle=cycle, selected_forward=chosen,
                         weighted_best={k: v['raw']*v['scale'] for k, v in cycle['best_scores'].items()},
                         weighted_forward={k: v*scales[k] for k, v in chosen['scores'].items() if k in scales} if chosen else None,
                         forward_definition=forward_note,
                         forward_is_best_turning_candidate=bool(chosen and not forward))
        evidence.append(entry)
    return evidence


def runtime_validation(work):
    evidence = []
    files = list((work/'sweep').glob('*/controller_server_runtime.yaml'))
    files += list(work.glob('*/controllers/controller_server_runtime.yaml'))
    for file in sorted(files):
        actual = yaml.safe_load(file.read_text())['/controller_server']['ros__parameters']
        costmap_file = file.parent/'local_costmap/local_costmap_runtime.yaml'
        if not costmap_file.exists():
            continue
        costmap = yaml.safe_load(costmap_file.read_text())['/local_costmap/local_costmap']['ros__parameters']
        entry = dict(file=str(file.relative_to(work)), GoalAlign_scale=actual['DWB.GoalAlign.scale'],
                     GoalDist_scale=actual['DWB.GoalDist.scale'],
                     PathOrientation_scale=actual['DWB.PathOrientationCritic.scale'],
                     PathPosition_scale=actual['DWB.PathPositionCritic.scale'],
                     sim_time_s=actual['DWB.sim_time'], local_costmap_resolution_m=costmap['resolution'])
        assert entry['GoalAlign_scale'] == 0. and entry['PathPosition_scale'] == 64.
        assert entry['local_costmap_resolution_m'] == .01
        assert 'GoalAlign' in actual['DWB.critics']
        assert actual['DWB.trajectory_generator_name'] == 'dwb_plugins::LimitedAccelGenerator'
        assert [actual['DWB.'+k] for k in ('vx_samples', 'vy_samples', 'vtheta_samples')] == [7, 7, 7]
        assert actual['DWB.sim_period'] == 1/30 and actual['DWB.max_speed_xy'] == .22
        evidence.append(entry)
    assert len(evidence) >= 10
    return evidence


def validate(work, output):
    initial = base.read_json(work/'initial_source_hashes.json')
    protected = base.read_json(work/'initial_protected_hashes.json')
    files = [p for name in ('params', 'scripts', 'tests', 'src') for p in (ROOT/name).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    owned = {'scripts/dwvp_access_e2.py', 'params/dwvp_access_e2_route_design_v5.yaml',
             'scripts/dwvp_access_e2_v5_run.py', 'scripts/dwvp_access_e2_v5_report.py', 'tests/test_access_e2_v5.py'}
    observed = dict(changed=[p for p in initial if p in hashes and hashes[p] != initial[p]],
                    new=[p for p in hashes if p not in initial], missing=[p for p in initial if p not in hashes])
    changes = {key: [p for p in values if p in owned] for key, values in observed.items()}
    changes['other_observed_changes'] = {key: [p for p in values if p not in owned] for key, values in observed.items()}
    save(output/'changed_files.json', changes); save(output/'source_manifest.json', hashes)
    for name in changes['changed']+changes['new']:
        destination = output/'implementation'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    old = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v4.yaml').read_text())
    new = yaml.safe_load((work/'selected_config.yaml').read_text())
    saved = base.read_json(work/'initial_E1_rendered.json')
    params = ROOT/'params/hsrb_dwvp_access_params.yaml'
    rendered = {c: render_parameters(params, new, c) for c in saved}
    old_e2, new_e2 = [render_parameters(params, c, 'E2_environment') for c in (old, new)]
    git = base.read_json(work/'initial_git.json')
    git_now = {c: subprocess.check_output(['git', *c.split()], cwd=ROOT, text=True)
               for c in ('rev-parse HEAD', 'branch --show-current')}
    save(output/'git_observations.json', dict(initial={c: git[c] for c in git_now}, current=git_now,
        note='This task only ran read-only git commands. Concurrent HEAD changes are recorded and preserved.'))
    protected_changes = [dict(file=p, initial_sha256=sha,
        current_sha256=digest(Path(p)) if Path(p).is_file() else None)
        for p, sha in protected.items() if not Path(p).is_file() or digest(Path(p)) != sha]
    unchanged_protected = not protected_changes
    save(output/'protected_file_changes_observed.json', dict(changes=protected_changes,
        note='These protected files were only read by this task. Concurrent differences are recorded and preserved, never reverted.'))
    snapshots = runtime_validation(work); save(output/'runtime_parameter_checks.json', snapshots)
    test_match = re.search(r'(\d+) passed', (work/'tooling-test.log').read_text())
    validation = dict(E1_rendered_equal_to_initial=rendered == saved,
        E1_conditions_equal_to_v4=all(old['conditions'][c] == new['conditions'][c] for c in saved),
        existing_parameter_files_unchanged=all(hashes.get(p) == sha for p, sha in initial.items() if p.startswith('params/')),
        protected_paper_files_unchanged=unchanged_protected, protected_file_count=len(protected),
        protected_paper_changed_file_count=len(protected_changes),
        protected_map_data_and_prior_reports_unchanged=all(Path(p['file']).is_relative_to(PAPER/'manuscript') for p in protected_changes),
        route_csv_and_metadata_byte_identical=all(digest(work/'route'/f) == digest(V4/'B_graph/route'/f)
                                                for f in ('E2_environment.csv', 'path_metadata.json')),
        local_and_global_costmap_equal_to_v4=all(old_e2[k] == new_e2[k] for k in ('local_costmap', 'global_costmap')),
        other_controllers_equal_to_v4=all(old_e2['controller_server']['ros__parameters'][c] ==
            new_e2['controller_server']['ros__parameters'][c] for c in ('VP_CLIP', 'DWVP', 'MPPI')),
        git_branch_unchanged=git_now['branch --show-current'] == git['branch --show-current'],
        git_head_unchanged=git_now['rev-parse HEAD'] == git['rev-parse HEAD'],
        runtime_snapshots_verified=len(snapshots), tooling_tests_passed=int(test_match[1]) if test_match else None,
        physical_trials=0)
    assert all(value is True for key, value in validation.items() if key not in
               ('protected_file_count', 'protected_paper_files_unchanged', 'protected_paper_changed_file_count', 'git_head_unchanged',
                'runtime_snapshots_verified', 'tooling_tests_passed', 'physical_trials'))
    assert validation['tooling_tests_passed']
    save(output/'validation.json', validation)
    return validation, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); w, destination = args.work.resolve(), args.output.resolve()
    if not w.is_relative_to('/tmp') or w == Path('/tmp'):
        parser.error('--work must be under /tmp')
    if destination != DEST and not (destination.is_relative_to('/tmp') and destination != Path('/tmp')):
        parser.error('Report destination is outside the authorized paths')
    if destination.exists():
        parser.error('Report destination already exists')
    diag = diagnose(w)
    o = w/('report_export_'+str(time.time_ns())); o.mkdir()
    base.METHODS = ['VP_CLIP', 'DWVP', 'MPPI', 'DWB']
    base.LABELS['DWVP'] = 'DWVP'
    if 'straight_cruise_mean_speed_m_s' not in base.METRICS:
        base.METRICS.append('straight_cruise_mean_speed_m_s')
    facts = base.export_route('B_graph', w/'route', w/'check_v4', w/'sweep', w/'verify', w, o)
    sweep = base.read_json(w/'sweep/report.json')
    selection = base.read_json(w/'selection.json')
    rows = base.read_json(w/'verify/report.json')['rows']
    path = np.loadtxt(w/'route/E2_environment.csv', delimiter=',', skiprows=1)
    meta = base.read_json(w/'route/path_metadata.json')
    sweep_figure(sweep['rows'], w, path, meta, o)
    final_table = result_table(facts['groups'])
    sweep_md = sweep_table(sweep['rows'])
    (o/'results_table.md').write_text(final_table+'\n')
    (o/'B_graph/results_table.md').write_text(final_table+'\n')
    (o/'sweep_table.md').write_text(sweep_md+'\n')
    repeat_stats = []
    for entry in selection['validations']:
        report = base.read_json(w/entry['report'])
        stats = facing_stats(report['rows'], w, path, meta)
        summary = {}
        for key in ('mean_abs_heading_error_deg', 'max_abs_heading_error_deg'):
            values = [r[key] for r in stats if r[key] is not None]
            summary[key] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                maximum=float(max(values)) if values else None)
        repeat_stats.append(dict(**entry, facing=stats, summary=summary))
    final_facing = facing_stats(rows, w, path, meta)
    save(o/'facing_statistics.json', dict(selection_repeats=repeat_stats, final_verify=final_facing,
        definition='Absolute wrapped yaw error at XY-projected arc length inside the fixed facing interval only; sample mean and sample maximum, excluding transition intervals.'))
    save(o/'dwb_diagnosis.json', diag)
    repeat_lines = ['| 候補 | 反復 | 完走 | 正対区間平均 [deg] | 正対区間最大 [deg] | 正対区間通過 |',
                    '|---|---:|---:|---:|---:|---|']
    for entry in repeat_stats:
        for r in entry['facing']:
            repeat_lines.append('| '+' | '.join([entry['candidate'], str(r['repeat']), '1/1' if r['success'] else '0/1',
                number(r['mean_abs_heading_error_deg']), number(r['max_abs_heading_error_deg']),
                '全区間' if r['full_facing_interval_traversed'] else '部分/未到達'])+' |')
    repeat_md = '\n'.join(repeat_lines); (o/'selection_repeats_table.md').write_text(repeat_md+'\n')
    diagnostic_lines = []
    for entry in diag:
        diagnostic_lines.append(f"{entry['stage']}: 5秒の購読窓で{entry['captured_cycles']}周期を保存。診断試行の完走={entry['success']}。")
        if entry['retried_empty_capture']:
            diagnostic_lines.append('初回はROS時刻差から決めた購読開始が遅く評価記録が0件だったため、単調時計の試行上限に余裕を持たせて別試行で再取得した。初回の生データも保存した。')
        failure = entry['sweep_failure']
        diagnostic_lines.append(f"掃引の未完走条件{failure['candidate']}（{failure['status']}）の終了位置(x,y,yaw)={failure['final_pose']}、"
            f"投影弧長={failure['final_projected_s_m']:.4f} m、最後の5秒間の変位={failure['final_5s_displacement_m']:.4f} m。")
        if failure['final_5s_displacement_m'] >= .01:
            diagnostic_lines.append('終了直前にも移動を継続していた。')
        elif failure['final_5s_displacement_m'] < .001:
            diagnostic_lines.append('終了直前の並進はほぼ停止していた（5秒間の変位が1 mm未満）。')
        if 'cycle' not in entry:
            diagnostic_lines.append('評価メッセージを取得できず、得点比較は未確定。')
            continue
        cycle = entry['cycle']; forward = entry['selected_forward']
        diagnostic_lines.append(f"報告する1周期の位置(x,y,yaw)={cycle['pose']}。最良速度={cycle['best_velocity']}、"
            f"合計={cycle['best_total']:.6f}、Oscillation棄却={cycle['rejected_by_critic'].get('Oscillation',0)}/{cycle['candidates']}候補。")
        if forward:
            diagnostic_lines.append(f"前進候補の速度(vx,vy,wz)={forward['velocity']}、記録合計={forward['total']:.6f}。" +
                ('この周期にはvx>0・vy=wz=0の候補がなく、最良候補自体が旋回を伴う前進候補だったため、両列は同じ候補を示す。'
                 if entry['forward_is_best_turning_candidate'] else
                 '前進はv4と同じくvx>0・vy=wz=0の候補とし、最後のPathPositionCriticまで採点された候補を優先した。'))
        diagnostic_lines += ['','| critic | 最良のraw × scale = 得点 | 前進のraw × scale = 得点 |', '|---|---:|---:|']
        for name in ('GoalDist', 'PathOrientationCritic', 'PathPositionCritic', 'Oscillation'):
            best = cycle['best_scores'].get(name)
            b = f"{best['raw']:.6f} × {best['scale']:.6f} = {best['raw']*best['scale']:.6f}" if best else '記録なし'
            f = (f"{forward['scores'][name]:.6f} × {best['scale']:.6f} = {forward['scores'][name]*best['scale']:.6f}"
                 if forward and best and name in forward['scores'] else '未評価/記録なし')
            diagnostic_lines.append(f'| {name} | {b} | {f} |')
        diagnostic_lines.append('')
    diagnostic_text = '\n'.join(diagnostic_lines) if diag else '8条件とも完走したため、追加の評価記録は不要だった。'
    for file in w.glob('*.log'):
        if file.name != 'pipeline.log':
            shutil.copy2(file, o/file.name)
    for file in ('initial_source_hashes.json', 'initial_protected_hashes.json', 'initial_E1_rendered.json',
                 'initial_git.json', 'plan_before_results.json', 'selection.json', 'selected_config.yaml', 'commands.jsonl'):
        shutil.copy2(w/file, o/file)
    shutil.copy2(w/'sweep/sweep.csv', o/'sweep.csv')
    for entry in selection['validations']:
        folder = w/Path(entry['report']).parent
        with tarfile.open(o/(folder.name+'_raw.tar.gz'), 'w:gz') as archive:
            archive.add(folder, arcname=folder.name)
        shutil.copy2(folder/'report.json', o/(folder.name+'_report.json'))
    for folder in sorted(w.glob('diagnostic_goal_*')):
        if folder.is_dir():
            shutil.copytree(folder, o/folder.name)
    old_rows = base.read_json(V4/'B_graph/verify_report.json')['rows']
    old_group = next(g for g in base.statistics(old_rows) if g['method'] == 'DWB')
    new_group = next(g for g in facts['groups'] if g['method'] == 'DWB')
    save(o/'v4_dwb_comparison.json', dict(v4=old_group, v5=new_group,
        v4_settings=dict(GoalAlign_scale=24, GoalDist_scale=24, PathOrientation_scale=4,
                         PathPosition_scale=64, sim_time_s=2), separate_session=True))
    comparison = (f"v4 DWB（GoalAlign 24 / 方位4 / sim_time 2 s）→v5: 完走 {old_group['successes']}/3→{new_group['successes']}/3、"
        f"方位平均 {old_group['eval_mean_heading_error_deg']['mean']:.2f}→{new_group['eval_mean_heading_error_deg']['mean']:.2f} deg、"
        f"位置平均 {old_group['eval_mean_position_error_m']['mean']:.4f}→{new_group['eval_mean_position_error_m']['mean']:.4f} m（別セッション）。")
    validation, changes = validate(w, o)
    validation['sweep_plan_written_before_first_trial'] = (w/'sweep/plan_before_results.json').stat().st_mtime_ns < min(
        (w/'sweep'/r['candidate']/'results.json').stat().st_mtime_ns for r in sweep['rows'])
    assert validation['sweep_plan_written_before_first_trial']
    save(o/'validation.json', validation)
    for file in ('scripts/dwvp_access_e2_v5_run.py', 'scripts/dwvp_access_e2_v5_report.py',
                 'scripts/dwvp_access_e2.py', 'tests/test_access_e2_v5.py', 'params/dwvp_access_e2_route_design_v5.yaml'):
        dest = o/'implementation'/file; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/file, dest)
    reproduce = ('#!/usr/bin/env bash\nset -euo pipefail\n'
        'work=${DWVP_E2_RERUN_WORK:-/tmp/dwvp-e2-v5-rerun-$(date +%Y%m%d-%H%M%S)}\n'
        'output=${DWVP_E2_RERUN_OUTPUT:-${work}-report}\n'
        'export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dwvp-e2-v5-mpl\n'
        'python3 '+shlex.quote(str(ROOT/'scripts/dwvp_access_e2_v5_run.py'))+' --work "$work" --output "$output"\n')
    (o/'reproduce.sh').write_text(reproduce)
    selected = selection['selected']; settings = selected['overrides']
    winner_text = (f"{selected['candidate']}（GoalDist.scale={settings['GoalDist.scale']:g}、"
        f"PathOrientationCritic.scale={settings['PathOrientationCritic.scale']:g}、sim_time={settings['sim_time']:g} s）")
    errors = [dict(controller=r['controller'], repeat=r['repeat'], success=r['success'],
                   data_errors=r.get('data_errors'), failed_compute_calls=r.get('timing',{}).get('failed_calls'),
                   recorder_error=r.get('recorder_error')) for r in rows]
    save(o/'trial_quality.json', errors)
    readme = f'''# E2 経路v4・DWB設定v5の合成実験

選定条件は{winner_text}。3/3完走した選定確認候補の有無: {selection['robust_selection']}。4手法の合成実験は同じ今回のセッションで各3反復を新規に実行した。物理試行は0件。

## 設定の差分

経路はv4のCSVとmetadataをバイト単位で再利用した。waypointは(2.5,0.05)→(−0.05,0.05)→(−0.05,−2.6)→(2.5,−2.6)、円弧半径0.3 m、正対区間(−0.05,−0.8)–(−0.05,−1.8)、yaw=π、遷移長0.35 m。経路の再生成とcheck-pathの再実行はしていない。`B_graph/check/`はv4の検査結果の写し。

`params/dwvp_access_e2_route_design_v5.yaml`でDWBのGoalAlign.scaleを0にした。critic一覧は維持した。GoalDist.scale∈{{24,96}}、PathOrientationCritic.scale∈{{16,64}}、sim_time∈{{2,3}} sの直積を掃引する。PathPositionCritic.scale=64、両独自criticのaggregation=mean、LimitedAccelGenerator、sim_period=1/30 s、標本7/7/7、max_speed_xy=0.22 m/s、Oscillation・BaseObstacle・ObstacleFootprint・RotateToGoalはv4のまま。

全手法のlocal costmap分解能は0.01 m。DWVPはτ_y=0.6 s、L=0.11 m固定。MPPIはPathAngle=6、PathAlign=30、wz_std=0.2を維持し再掃引していない。VP_CLIPもv4と同じ展開設定。実行時snapshot {validation['runtime_snapshots_verified']}組でGoalAlign.scale=0と分解能0.01 m、固定したDWB設定を照合した（`runtime_parameter_checks.json`）。

## 8条件の掃引と選定

`plan_before_results.json`と`sweep_plan_before_results.json`に、結果取得前に候補と規則を保存した（後者は`B_graph/`内）。各1試行、逐次実行。完走した候補の位置・方位誤差時間積分を、それぞれ完走候補内の最小値で割り、和が最小の条件を選ぶ。同点は表の順。ゼロ最小値の扱いも計画に固定した。失敗候補の積分・巡航は部分軌道の観測値。

{sweep_md}

掃引の首位: {selection['sweep_winner']['candidate']}。最終選定: {winner_text}。詳細は`selection.json`と`selected_config.yaml`。

## 停滞の確認

{diagnostic_text}

診断を行った場合は選定・計算時間の集計から除外した。`dwb_diagnosis.json`に選んだ1周期と得点を保存した。DWBの通常設定では候補の合計が既知の最良値を超えると採点を打ち切るため、後続criticが未評価の候補はその旨を示す。

## 選定条件の3反復

掃引首位を3反復し、1件でも未完走なら完走候補内の次点も3反復する規則を実行した。最大2条件。正対区間の値はXY投影弧長が固定yaw区間内にあるサンプルの絶対方位誤差で、遷移区間を除く。

{repeat_md}

## 4手法の新規3反復

±は3反復の標本SD（ddof=1）。位置・方位誤差は共通評価区間s∈[0,経路長−0.6 m]。平均・最大を各試行で求めてから3試行を集計した。走行時間はFollowPath完走まで。直線巡航速度はs∈[0.3,経路長−0.6 m]かつ参照XY接線の微分の絶対値≤0.1 rad/mのodometry速度ノルム平均で、直線上の正対遷移も含む。

{final_table}

計算時間はTimedControllerが内側pluginのcomputeVelocityCommandsを囲んだ値で、DWBのprepareと候補採点を含み、costmapの別スレッド処理を含まない。各試行の平均計算時間を3反復で集計した。未完走の走行時間は欠測、誤差と巡航の有限値は部分観測として扱う。試行のdata_errors、recorder_error、失敗したplugin呼び出しは`trial_quality.json`で確認できる。

走行時間と誤差積分は記録されたROS時刻に基づき、120秒の試行打ち切りは単調時計で判定する。この実行では両時計の経過時間が一致せず、timeout試行の記録時間は120秒を超えた。時計差の原因は特定していない。記録走行時間から単調時計の制限までの余裕を直接計算しない。

{comparison}

## 図・生データ・再実行

- `B_graph/map_tracking.png`: 地図、v4経路、正対区間と遷移、4手法の各3反復の追従経路。
- `B_graph/heading_speed.png`: 各手法3反復の弧長に対する絶対方位誤差と並進速度。
- `dwb_sweep_heading_speed.png`: 掃引8条件の方位誤差と速度の弧長プロファイルを1枚に配置。
- `B_graph/sweep_raw.tar.gz`、`B_graph/verify_raw.tar.gz`: 全試行のCSV、展開設定、runtime snapshot、結果判定、timing、ログ。元の作業先は`{w}`。アーカイブ内の`/work/...`はこのディレクトリに対応する。
- `selected_repeat_*_raw.tar.gz`: 選定確認の独立した3反復。`facing_statistics.json`に正対区間の各反復の値を保存。
- `bash reproduce.sh`: 新規の/tmp作業先にbuildから実験・図表まで再実行。既存のローカルDocker imageと依存を使用し、Docker呼び出しはすべて`e2_tooling.sh`の`--network none`経由。経路入力はv4成果物。`implementation/`に今回の設定とスクリプトを保存。

## 検証

`scripts/e2_tooling.sh test`: {validation['tooling_tests_passed']} passed。E1設定と作業前の全E1展開結果、既存paramsファイル（v4を含む）、map・data・既存v1〜v4成果物、v4のlocal/global costmap、VP_CLIP・DWVP・MPPIの展開設定の不変を確認した（`validation.json`）。ネットワーク、サブエージェント、物理ロボットへの接続、commit、push、branch切替、作業ツリー削除は行っていない。

原稿側の保護対象{validation['protected_file_count']}ファイルを開始時のハッシュと照合し、{validation['protected_paper_changed_file_count']}ファイルに差分を検出した（`protected_file_changes_observed.json`）。この作業では保護対象への書き込みを行っていない。作業中に観測した他の変更は保持し、元に戻す操作も行っていない。

branchは開始時と同じ。HEADの開始時との一致は{validation['git_head_unchanged']}で、前後の値を`git_observations.json`に保存した。この作業で実行したGitコマンドは読み取り専用だけである。パッケージ内で観測した今回の作業対象以外の変更も、`changed_files.json`の`other_observed_changes`に分けて記録した。

## 未確定事項

この結果は静的地図と理想的なray scanを用いた運動学的な合成試行である。物理HSRでの通行可否、測位・駆動誤差、人への不快感は未評価。元地図は0.05 mであり、local costmapの0.01 m化で地図の幾何精度が向上したとは判断しない。MPPIの乱数と実時間スケジューリングで数値は変動する。各1試行の選定と3反復から統計的優越性は判断できない。DWBの並進ノルム上限0.22 m/sと他手法の各軸上限±0.22 m/sは、斜行時の上限が異なる。v4比較は別セッションであり、GoalAlignだけの単独効果を示す実験ではない。
'''
    (o/'README.md').write_text(readme)
    save(o/'raw_locations.json', dict(work=str(w), route_source=str(V4/'B_graph/route'),
                                    sweep=str(w/'sweep'), verify=str(w/'verify')))
    save(o/'SHA256SUMS.json', {str(p.relative_to(o)): digest(p) for p in o.rglob('*') if p.is_file()})
    shutil.copytree(o, destination)
    print(sweep_md); print(repeat_md); print(final_table)
    print('Report: '+str(destination))


if __name__ == '__main__':
    main()
