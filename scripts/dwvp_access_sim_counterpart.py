#!/usr/bin/env python3
"""Simulate E1a on frozen/recorded paths using the external locked simulator.

Run from the simulator checkout with PYTHONDONTWRITEBYTECODE=1 and
uv run --offline --locked --python 3.11.11 --no-sync /path/to/this/script ...
Only --output is written. Planned paths require explicit --include-planned.
"""
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

import dwvp_access_experiment as experiment
from dwvp_access_metrics import lateral_response_metrics, tracking_metrics

METRICS = ('lateral_convergence_time_s', 'settling_2pct_time_s',
           'settling_5pct_time_s', 'tau_fit_s', 'crossing_m')


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def simulate_session(session, output, *, include_planned=False):
    from omnidirectional_dwvp.config import Config
    from omnidirectional_dwvp.simulation import simulate
    from omnidirectional_dwvp.access_v3_metrics import fitted_time_constant
    import omnidirectional_dwvp.config as simulator_module

    session, output = Path(session).resolve(), Path(output).resolve()
    if output == session or session in output.parents:
        raise ValueError('Counterpart output must be outside the source session')
    manifest = json.loads((session/'manifest.json').read_text())
    settings = yaml.safe_load((session/manifest['config_file']).read_text())
    selected = [t for t in manifest['trials'] if t['task'] in experiment.RESPONSE_CONDITIONS
                and t['controller'] == 'DWVP'
                and (include_planned or (session/'runs'/t['id']/'reference.csv').exists())]
    if not selected:
        raise ValueError('No recorded E1a paths; use --include-planned only for prospective checks')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for trial in selected:
        _, _, frozen_path = experiment.load_trial(session, trial['id'])
        folder = session/'runs'/trial['id']
        if (folder/'reference.csv').exists():
            initial, path, policy = experiment.recorded_geometry(session, trial, folder)
            source = 'recorded'
        else:
            initial, path, policy = experiment.start_pose(manifest, trial), frozen_path, 'session_fixed'
            source = 'planned'
        condition = settings['conditions'][trial['task']]
        common = experiment.condition_common(settings, trial['task'])
        frozen = yaml.safe_load((session/trial['params_file']).read_text())
        controller = frozen['controller_server']['ros__parameters']['DWVP']
        speed = controller['vp_translation_speed']
        if speed <= 0:
            speed = min(controller['max_vel_x'], -controller['min_vel_x'],
                        controller['max_vel_y'], -controller['min_vel_y'])
        look, tau = controller['min_lookahead_dist'], controller['lateral_time_constant']
        if (controller['max_lookahead_dist'] != look or tau <= 0
                or not np.isclose(speed, .22)):
            raise ValueError('E1a requires fixed lookahead, positive tau and v_vp=0.22 m/s')
        if not np.isclose(experiment.arclength(path[:, :2])[-1], 2.5):
            raise ValueError('E1a counterpart requires the recorded 2.5 m reference')
        cfg = Config(frequency=common['control_frequency_hz'], timeout=settings['trial']['timeout_s'],
            vx_min=controller['min_vel_x'], vx_max=controller['max_vel_x'],
            vy_min=controller['min_vel_y'], vy_max=controller['max_vel_y'],
            w_min=controller['min_vel_theta'], w_max=controller['max_vel_theta'],
            ax=controller['max_accel_x'], ay=controller['max_accel_y'], aw=controller['max_accel_theta'],
            desired_linear_vel=controller['desired_linear_vel'], vp_translation_speed=speed,
            fixed_lookahead=look, lookahead_min=look, lookahead_max=look,
            lookahead_time=controller['lookahead_time'], response_distance=tau*.22,
            orientation_time_weight=controller['orientation_time_weight'],
            min_orientation_time=controller['min_orientation_time'],
            approach_distance=controller['approach_velocity_scaling_dist'],
            min_approach_speed=controller['min_approach_linear_velocity'])
        result = simulate(path, 'dwvp', cfg, initial_pose=initial, seed=0)
        a = result.arrays
        direction = path[-1, :2]-path[0, :2]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        errors = (a['poses'][:, :2]-path[0, :2]) @ normal
        response = lateral_response_metrics(a['times'], errors, complete=result.metrics['success'])
        # Independent simulator Test 1 fit checks the shared analysis contract.
        oracle = fitted_time_constant(a['times'], errors)
        assert (oracle is None and response['tau_fit_s'] is None) or np.isclose(oracle, response['tau_fit_s'], atol=1e-9, rtol=0)
        tracking, _, _ = tracking_metrics(a['times'], a['poses'], path, condition, initial)
        e0 = float((initial[:2]-path[0, :2]) @ normal)
        hits = np.flatnonzero(np.abs(errors) <= abs(e0)*settings['metrics']['lateral_convergence_ratio'])
        row = dict(trial_id=trial['id'], task=trial['task'], controller='DWVP',
                   direction=trial.get('direction', 'forward'), source=source, reference_policy=policy,
                   fixed_lookahead_m=look, lateral_time_constant_s=tau, response_distance_m=tau*.22,
                   success=result.metrics['success'], duration_s=result.metrics['duration_s'],
                   lateral_convergence_time_s=float(a['times'][hits[0]]) if len(hits) else None,
                   crossing_m=tracking['crossing_m'], **response)
        destination = output/trial['id']
        destination.mkdir()
        np.savez_compressed(destination/'trajectory.npz', **a)
        np.savetxt(destination/'reference.csv', path, delimiter=',', header='x,y,yaw', comments='')
        experiment.write_json(destination/'inputs.json', dict(config=asdict(cfg), initial_pose=initial.tolist(),
            reference_sha256=experiment.digest(destination/'reference.csv'), source=source))
        rows.append(row)
    write_csv(output/'simulation_metrics.csv', rows)
    robot = []
    if (session/'trial_metrics.csv').exists():
        with (session/'trial_metrics.csv').open() as stream:
            robot = list(csv.DictReader(stream))
    comparison = []
    for task, direction in sorted({(r['task'], r['direction']) for r in rows}):
        simulations = [r for r in rows if (r['task'], r['direction']) == (task, direction)]
        ids = {r['trial_id'] for r in simulations if r['source'] == 'recorded'}
        observed = [r for r in robot if r['trial_id'] in ids]
        item = dict(task=task, controller='DWVP', direction=direction,
                    recorded_paths=sum(r['source'] == 'recorded' for r in simulations),
                    planned_paths=sum(r['source'] == 'planned' for r in simulations))
        for metric in METRICS:
            real = [float(r[metric]) for r in observed if r.get(metric) and np.isfinite(float(r[metric]))]
            sim = [r[metric] for r in simulations if r[metric] is not None]
            item.update({metric+'_robot_n': len(real),
                         metric+'_robot_mean': float(np.mean(real)) if real else None,
                         metric+'_robot_sd': float(np.std(real, ddof=1)) if len(real)>1 else None,
                         metric+'_simulation_n': len(sim),
                         metric+'_simulation_mean': float(np.mean(sim)) if sim else None,
                         metric+'_simulation_min': min(sim) if sim else None,
                         metric+'_simulation_max': max(sim) if sim else None})
        comparison.append(item)
    write_csv(output/'comparison.csv', comparison)
    source_root = Path(simulator_module.__file__).resolve().parents[2]
    provenance = dict(purpose='Deterministic counterparts; no physical trials executed',
        source_session=str(session), source_manifest_sha256=experiment.digest(session/'manifest.json'),
        simulator_commit=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),
        simulator_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(simulator_module.__file__).parent.glob('*.py')},
        uv_lock_sha256=experiment.digest(source_root/'uv.lock'), python=sys.version,
        note='Robot means retain finite observations from all attempts, as in the session summary. '
             'Simulation means/ranges describe deterministic counterparts of the selected paths, not independent repeats. '
             'No obstacles or localization/dynamic noise. Simulator goal tolerance is 0.02 m / 1 degree with velocity convergence; '
             'HSR uses 0.1 m / 0.3 rad. Response settling is action-wide; crossing uses the frozen spatial window.',
        trials=rows, comparison=comparison)
    experiment.write_json(output/'report.json', provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-planned', action='store_true')
    args = parser.parse_args()
    report = simulate_session(args.session, args.output, include_planned=args.include_planned)
    print(f"Simulated {len(report['trials'])} E1a paths: {args.output}")


if __name__ == '__main__':
    main()
