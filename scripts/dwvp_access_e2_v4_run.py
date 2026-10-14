#!/usr/bin/env python3
"""Reproduce route v4 using local inputs and e2_tooling.sh --network none only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from dwvp_access_e2 import merge, RULE, STRAIGHT_RULE
from dwvp_access_experiment import render_parameters
from dwvp_access_path import rounded_graph, load_map, clearance_profile

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path('/home/decwest/decwest_workspace/tex_docker_environment/projects/DWVP_ACCESS')


def write_json(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map', type=Path, default=PAPER/'map/map.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume completed stages in an existing work directory')
    args = parser.parse_args()
    w, output = args.work.resolve(), args.output.resolve()
    if not w.is_relative_to('/tmp') or w == Path('/tmp'):
        parser.error('--work must be a subdirectory of /tmp')
    if not (output.is_relative_to('/tmp') or output == PAPER/'docs/hardware/e2_design_20261014_v4'):
        parser.error('Report destination is outside the authorized paths')
    if output.exists() or (w.exists() and not args.resume):
        parser.error('Use unused work/report paths, or --resume with an unused report path')
    w.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DWVP_E2_WORK=str(w), DWVP_E2_INPUT=str(args.map.parent.resolve()),
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/dwvp-e2-v4-mpl')
    commands = w/'commands.jsonl'

    def run(label, arguments, artifact=None, expected_rows=None):
        if artifact and (w/artifact).exists() and args.resume:
            print('Retain completed stage: '+label, flush=True)
            return
        command = [str(ROOT/'scripts/e2_tooling.sh'), *map(str, arguments)]
        with commands.open('a') as log:
            log.write(json.dumps(dict(stage=label, argv=command))+'\n')
        print('Run: '+label, flush=True)
        with (w/(label+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if expected_rows is not None and artifact and (w/artifact).exists():
            rows = json.loads((w/artifact).read_text())['rows']
            if len(rows) != expected_rows or any(r.get('setup_error') for r in rows):
                raise RuntimeError(label+': missing trial results or setup failure')
            # A non-completion is an experimental result, not a missing artifact.
        elif result.returncode:
            raise RuntimeError(label+': see '+str(w/(label+'.log')))
        if artifact and not (w/artifact).exists():
            raise RuntimeError(label+': output artifact missing')
        print('Finished: '+label, flush=True)

    base = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v4.yaml').read_text())
    initial = w/'initial_source_hashes.json'
    if not initial.exists():
        files = [p for folder in ('params', 'scripts', 'tests', 'src') for p in (ROOT/folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts]
        write_json(initial, {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        write_json(w/'initial_E1_rendered.json', {c: render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', base, c)
            for c in base['conditions'] if c.startswith('E1')})
    plan_file = w/'protocol_before_sweep.json'
    if not plan_file.exists():
        write_json(plan_file, dict(selection_rule=RULE, straight_cruise_rule=STRAIGHT_RULE,
            resolution_trials=dict(A=.05, B=.01, conditional_C=.02), repeats=3,
            selected_v3_dwb=dict(sim_time=3., orientation_scale=16., position_scale=64.),
            sweep=dict(sim_time=[2.,3.], orientation_scale=[4.,16.], position_scale=[16.,64.],
                       repeats=1, workers=1, resolution=.01, samples=[7,7,7]),
            mppi_fixed=dict(PathAngle=6., PathAlign=30., wz_std=.2),
            no_completed_candidate_policy='No winner; retain the predeclared v3 setting 3/16/64 for final comparison at 0.01 m',
            physical_trials=0))
    run('build', ['build'], 'ws/install/setup.bash')
    path_config = yaml.safe_load((ROOT/'params/dwvp_access_path_v4.yaml').read_text())
    info, pixels, blocked = load_map(args.map)
    trial_path, _ = rounded_graph([[2.5,.05],[-.05,.05],[-.05,-2.6],[2.5,-2.6]])
    distances, _ = clearance_profile(trial_path, info, blocked)
    near = np.linalg.norm(trial_path[:,:2]-[-.05,-2.6], axis=1) < .5
    lower = -2.6 if float(distances[near].min()) >= .3 else -2.5
    write_json(w/'geometry_decision.json', dict(lower_y_m=lower,
        clearance_near_requested_lower_corner_m=float(distances[near].min()), near_definition='within 0.5 m of (-0.05,-2.6)'))
    for name, resolution in [('001', .01), ('005', .05)]:
        config = copy.deepcopy(path_config); config['inspection_costmap_resolution_m'] = resolution
        (w/('path_'+name+'.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
        run('path'+name, ['path', '--graph', '--corner-radius', '.3', '--map', '/input/'+args.map.name,
            '--config', '/work/path_'+name+'.yaml', '--output', '/work/route_'+name,
            '--start', '2.5', '.05', str(np.pi), '--via', '-.05', '.05', '--via', '-.05', str(lower),
            '--goal', '2.5', str(lower), '0', '--facing', '-.05', '-.8', '-.05', '-1.8', str(np.pi),
            '--transition', '.35'], 'route_'+name+'/path_metadata.json')
        run('check'+name, ['check-path', '--path', '/work/route_'+name+'/E2_environment.csv',
            '--output', '/work/check_'+name], 'check_'+name+'/report.json')
    checks = json.loads((w/'check_001/report.json').read_text())
    if not all(g['at_least_0p1_m'] for g in checks['facing_corner_gaps']):
        raise RuntimeError('Facing endpoints need review: gaps below 0.1 m')
    if (w/'route_001/E2_environment.csv').read_bytes() != (w/'route_005/E2_environment.csv').read_bytes():
        raise RuntimeError('Inspection resolution changed the route')
    common = ['--path','/work/route_001/E2_environment.csv','--map','/input/'+args.map.name,
              '--grid-profile','route-v4']
    for label, resolution in [('A',.05),('B',.01),('C',.02)]:
        if label == 'C':
            previous = json.loads((w/'resolution_B/report.json').read_text())['rows']
            if all(r['success'] for r in previous):
                break
        config = copy.deepcopy(base)
        config['conditions']['E2_environment'].update(methods=['DWB'], local_costmap_resolution_m=resolution)
        (w/('resolution_'+label+'.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
        run('resolution_'+label, ['verify', *common, '--config','/work/resolution_'+label+'.yaml',
            '--output','/work/resolution_'+label,'--repeats','3'], 'resolution_'+label+'/report.json', 3)
    if (w/'resolution_C/report.json').exists():
        rows = json.loads((w/'resolution_C/report.json').read_text())['rows']
        if not all(r['success'] for r in rows):
            # Capture near the end of a failed B trial, using the preceding
            # run's duration to avoid continuously publishing full evaluations.
            b = json.loads((w/'resolution_B/report.json').read_text())['rows']
            row = next(r for r in b if not r['success'])
            session = w/Path(row['session']).relative_to('/work')
            track = np.genfromtxt(session/'runs'/row['trial_id']/'tracking.csv', delimiter=',', names=True)
            elapsed = float(track['stamp_s'][-1]-track['stamp_s'][0])
            delay = max(10., elapsed-3.)
            run('diagnostic_B', ['verify', *common, '--config','/work/resolution_B.yaml',
                '--output','/work/diagnostic_B','--repeats','1','--diagnose-dwb','--diagnostic-after',str(delay)],
                'diagnostic_B/report.json', 1)
    run('sweep', ['sweep', *common, '--config',str('/ws/src/dwpp_test_simulation/params/dwvp_access_e2_route_design_v4.yaml'),
        '--output','/work/sweep','--sweep-workers','1'], 'sweep/report.json', 8)
    selection = json.loads((w/'sweep/report.json').read_text())['selected']['DWB']
    selected = copy.deepcopy(base)
    if selection:
        merge(selected['conditions']['E2_environment']['controller_overrides']['DWB'], selection['overrides'])
    (w/'selected_config.yaml').write_text(yaml.safe_dump(selected, sort_keys=False))
    run('verify', ['verify', *common, '--config','/work/selected_config.yaml',
        '--output','/work/verify','--repeats','3'], 'verify/report.json', 12)
    run('tooling-test', ['test'])
    run('runtime-provenance', ['inspect','bash','-c',
        'dpkg-query -W ros-humble-nav2-costmap-2d ros-humble-nav2-mppi-controller ros-humble-dwb-core ros-humble-dwb-critics; '
        'sha256sum /opt/ros/humble/lib/libmppi_controller.so /opt/ros/humble/lib/libdwb_core.so; cat /proc/net/dev'])
    subprocess.run([sys.executable, str(ROOT/'scripts/dwvp_access_e2_v4_report.py'),
        '--work', str(w), '--output', str(output)], check=True, env=env, cwd=ROOT)
    print('Report: '+str(output), flush=True)


if __name__ == '__main__':
    main()
