#!/usr/bin/env python3
"""Run the fixed v5 protocol offline, reusing the published v4 route verbatim."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from dwvp_access_e2 import grid, merge, RULE, STRAIGHT_RULE
from dwvp_access_experiment import render_parameters

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path('/home/decwest/decwest_workspace/tex_docker_environment/projects/DWVP_ACCESS')
V4 = PAPER/'docs/hardware/e2_design_20261014_v4'
DEST = PAPER/'docs/hardware/e2_design_20261014_v5'


def save(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def baseline(work):
    if (work/'initial_source_hashes.json').exists():
        return
    files = [p for name in ('params', 'scripts', 'tests', 'src') for p in (ROOT/name).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    save(work/'initial_source_hashes.json', {str(p.relative_to(ROOT)): digest(p) for p in files})
    protected = [PAPER/name for name in ('manuscript', 'data', 'map')]
    protected += [p for p in (PAPER/'docs/hardware').glob('e2_design_20261014*') if p.is_dir() and p != DEST]
    save(work/'initial_protected_hashes.json', {str(p): digest(p) for d in protected for p in d.rglob('*') if p.is_file()})
    config = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v4.yaml').read_text())
    save(work/'initial_E1_rendered.json', {c: render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', config, c)
        for c in config['conditions'] if c.startswith('E1')})
    save(work/'initial_git.json', {cmd: subprocess.check_output(['git', *cmd.split()], cwd=ROOT, text=True)
        for cmd in ('status --short', 'rev-parse HEAD', 'branch --show-current')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    w, output = args.work.resolve(), args.output.resolve()
    if not w.is_relative_to('/tmp') or w == Path('/tmp'):
        parser.error('--work must be a subdirectory of /tmp')
    if output != DEST and not (output.is_relative_to('/tmp') and output != Path('/tmp')):
        parser.error('Report destination is outside the authorized paths')
    if output.exists() or (w.exists() and not args.resume):
        parser.error('Use unused work/report paths, or --resume with an unused report path')
    w.mkdir(parents=True, exist_ok=True)
    baseline(w)
    env = dict(os.environ, DWVP_E2_WORK=str(w), DWVP_E2_INPUT=str(V4/'B_graph/route'),
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/dwvp-e2-v5-mpl')
    base = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v5.yaml').read_text())
    if not (w/'plan_before_results.json').exists():
        save(w/'plan_before_results.json', dict(selection_rule=RULE, straight_cruise_rule=STRAIGHT_RULE,
            grid=[dict(controller=c, candidate=n, overrides=o) for c, n, o in grid('route-v5')],
            fixed=dict(GoalAlign_scale=0., PathPositionCritic_scale=64., resolution_m=.01,
                       samples=[7, 7, 7], sim_period_s=1/30, max_speed_xy=.22),
            selection_validation='Test the sweep winner 3 times; if any fail, test the next eligible candidate 3 times. Maximum two candidates. Use the first 3/3 candidate; if neither passes, retain the sweep winner and explicitly report no robust selection.',
            no_completed_candidate_policy='Stop with no selected DWB; never select a failed sweep candidate.',
            diagnostics='For each GoalDist scale with a failed sweep, diagnose the first failed candidate in grid order; subscribe for 5 s near the recorded stop. Exclude from selection and timing.',
            repeats=3, verify_methods=['VP_CLIP', 'DWVP', 'MPPI', 'DWB'],
            route_source=str(V4/'B_graph/route'), path_sha256=digest(V4/'B_graph/route/E2_environment.csv'),
            config_sha256=digest(ROOT/'params/dwvp_access_e2_route_design_v5.yaml'), physical_trials=0))

    def run(label, arguments, artifact=None, expected_rows=None):
        if artifact and (w/artifact).exists() and args.resume:
            if expected_rows is not None:
                rows = json.loads((w/artifact).read_text())['rows']
                if len(rows) != expected_rows or any(r.get('setup_error') for r in rows):
                    raise RuntimeError(label+': existing stage is incomplete')
            print('Retain completed stage: '+label, flush=True)
            return
        command = [str(ROOT/'scripts/e2_tooling.sh'), *map(str, arguments)]
        with (w/'commands.jsonl').open('a') as log:
            log.write(json.dumps(dict(stage=label, argv=command))+'\n')
        print('Run: '+label, flush=True)
        with (w/(label+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if expected_rows is not None and artifact and (w/artifact).exists():
            rows = json.loads((w/artifact).read_text())['rows']
            if len(rows) != expected_rows or any(r.get('setup_error') for r in rows):
                raise RuntimeError(label+': missing trial results or setup failure')
        elif result.returncode:
            raise RuntimeError(label+': see '+str(w/(label+'.log')))
        if artifact and not (w/artifact).exists():
            raise RuntimeError(label+': output artifact missing')
        print('Finished: '+label, flush=True)

    run('build', ['build'], 'ws/install/setup.bash')
    for target, source in [('route', V4/'B_graph/route'), ('check_v4', V4/'B_graph/check')]:
        if not (w/target).exists():
            shutil.copytree(source, w/target)
    for name in ('E2_environment.csv', 'path_metadata.json'):
        if digest(w/'route'/name) != digest(V4/'B_graph/route'/name):
            raise RuntimeError('The v4 route must remain byte-identical: '+name)
    common = ['--path', '/work/route/E2_environment.csv', '--map', '/input/map.yaml', '--grid-profile', 'route-v5']
    run('sweep', ['sweep', *common, '--config', '/ws/src/dwpp_test_simulation/params/dwvp_access_e2_route_design_v5.yaml',
        '--output', '/work/sweep', '--sweep-workers', '1'], 'sweep/report.json', 8)
    sweep = json.loads((w/'sweep/report.json').read_text())
    for goal in (24., 96.):
        failed = next((r for r in sweep['rows'] if not r['success'] and r['overrides']['GoalDist.scale'] == goal), None)
        if failed is None:
            continue
        config = copy.deepcopy(base)
        config['conditions']['E2_environment']['methods'] = ['DWB']
        merge(config['conditions']['E2_environment']['controller_overrides']['DWB'], failed['overrides'])
        label = f'diagnostic_goal_{goal:g}'
        (w/(label+'.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
        session = w/Path(failed['session']).relative_to('/work')
        tracking = np.genfromtxt(session/'runs'/failed['trial_id']/'tracking.csv', delimiter=',', names=True)
        # ROS receive stamps can span more than the monotonic trial timeout.
        # Keep the five-second diagnostic window before that timeout.
        delay = max(10., min(float(tracking['stamp_s'][-1]-tracking['stamp_s'][0])-3.,
                            config['trial']['timeout_s']-10.))
        run(label, ['verify', *common, '--config', '/work/'+label+'.yaml', '--output', '/work/'+label,
            '--repeats', '1', '--diagnose-dwb', '--diagnostic-after', str(delay)], label+'/report.json', 1)
    eligible = [r for r in sweep['rows'] if r['success'] and r.get('selection_score') is not None]
    eligible.sort(key=lambda r: r['selection_score'])
    if not eligible:
        raise RuntimeError('No completed sweep candidate: DWB selection unavailable')
    selected = eligible[0]
    validations = []
    for rank, row in enumerate(eligible[:2], 1):
        config = copy.deepcopy(base)
        config['conditions']['E2_environment']['methods'] = ['DWB']
        merge(config['conditions']['E2_environment']['controller_overrides']['DWB'], row['overrides'])
        label = f'selected_repeat_{rank}'
        (w/(label+'.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
        run(label, ['verify', *common, '--config', '/work/'+label+'.yaml', '--output', '/work/'+label,
            '--repeats', '3'], label+'/report.json', 3)
        rows = json.loads((w/label/'report.json').read_text())['rows']
        passed = all(r['success'] for r in rows)
        validations.append(dict(rank=rank, candidate=row['candidate'], successes=sum(r['success'] for r in rows),
                                repeats=3, report=label+'/report.json'))
        if passed:
            selected = row
            break
    save(w/'selection.json', dict(sweep_winner=sweep['selected']['DWB'], selected=selected,
        validations=validations, robust_selection=any(v['successes']==3 for v in validations)))
    config = copy.deepcopy(base)
    merge(config['conditions']['E2_environment']['controller_overrides']['DWB'], selected['overrides'])
    (w/'selected_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    run('verify', ['verify', *common, '--config', '/work/selected_config.yaml', '--output', '/work/verify',
        '--repeats', '3'], 'verify/report.json', 12)
    run('tooling-test', ['test'])
    run('runtime-provenance', ['inspect', 'bash', '-c',
        'dpkg-query -W ros-humble-nav2-costmap-2d ros-humble-nav2-mppi-controller ros-humble-dwb-core ros-humble-dwb-critics; '
        'sha256sum /opt/ros/humble/lib/libmppi_controller.so /opt/ros/humble/lib/libdwb_core.so; cat /proc/net/dev'])
    subprocess.run([sys.executable, str(ROOT/'scripts/dwvp_access_e2_v5_report.py'),
        '--work', str(w), '--output', str(output)], cwd=ROOT, env=env, check=True)
    print('Report: '+str(output), flush=True)


if __name__ == '__main__':
    main()
