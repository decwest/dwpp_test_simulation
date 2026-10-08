#!/usr/bin/env python3
"""Network-isolated E2 critic sweep and repeated controller/recorder verification.

Run from a source checkout with tests/ available, after sourcing the built workspace.
Uses the same kinematic plant, controller stack, recorder and metrics as the package
end-to-end check. Synthetic free-space sensing is not physical pilot replay.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import yaml

import dwvp_access_experiment as experiment
from dwvp_access_metrics import COMMON_METRICS, project_tracking

ROOT = Path(__file__).resolve().parents[1]
RULE = ('Per controller, among completed candidates with finite evaluation integrals, '
        'minimize I_position/min(I_position) + I_heading/min(I_heading). '
        'Minima use completed candidates in that controller grid; exact ties use grid order. '
        'Zero-minimum limit: zero contributes 1, positive contributes infinity. '
        'No completed candidate means no selection. Common evaluation: s=0 to length-0.6 m.')


def grid():
    result = []
    for weight in (6., 12., 20.):
        result.append(('MPPI', f'mppi_angle_{weight:g}', {
            'PathAlignCritic': {'use_path_orientations': True},
            'PathAngleCritic': {'cost_weight': weight}}))
    for distance in (.1, .325, .5):
        for scale in (32., 48.):
            result.append(('DWB', f'dwb_distance_{distance:g}_scale_{scale:g}', {
                'max_speed_xy': .22, 'PathAlign.forward_point_distance': distance,
                'GoalAlign.forward_point_distance': distance, 'PathAlign.scale': scale}))
    return result


def select(rows):
    selected = {}
    for controller in ('MPPI', 'DWB'):
        eligible = [r for r in rows if r['controller'] == controller and r.get('success') and
                    all(r.get(k) is not None and math.isfinite(r[k]) for k in
                        ('eval_position_error_integral_m_s', 'eval_heading_error_integral_deg_s'))]
        if not eligible:
            selected[controller] = None
            continue
        keys = ('eval_position_error_integral_m_s', 'eval_heading_error_integral_deg_s')
        best = {k: min(r[k] for r in eligible) for k in keys}
        def score(row):
            return sum(row[k]/best[k] if best[k] > 0 else (1. if row[k] == 0 else math.inf) for k in keys)
        for row in eligible:
            value = score(row)
            row['selection_score'] = value if math.isfinite(value) else None
            row['normalization_minima'] = best
        winner = min(eligible, key=score)
        selected[controller] = dict(candidate=winner['candidate'], overrides=winner['overrides'],
                                    score=winner['selection_score'], normalization_minima=best)
    return selected


def motion_diagnostics(folder, path):
    def read(file):
        with (folder/file).open() as stream:
            return list(csv.DictReader(stream))
    tracking = read('tracking.csv')
    valid = [r for r in tracking if np.isfinite([float(r[k]) for k in ('x','y','yaw','stamp_s')]).all()]
    if not valid:
        return {}
    poses = np.array([[float(r[k]) for k in ('x','y','yaw')] for r in valid])
    projection = project_tracking(poses, path)
    stamps = np.array([float(r['stamp_s']) for r in valid])
    length = experiment.arclength(path[:,:2])[-1]
    output = {}
    for stream in ('raw', 'applied', 'odom'):
        samples = read(stream+'.csv')
        t = np.array([float(r['stamp_s']) for r in samples])
        speed = np.array([math.hypot(float(r['vx']),float(r['vy'])) for r in samples])
        omega = np.array([float(r['omega']) for r in samples])
        progress = np.interp(t, stamps, projection[:,2])
        # Omit initial acceleration distance and the unchanged goal approach zone.
        mask = (progress >= .3) & (progress <= length-.6) & (t >= stamps[0]) & (t <= stamps[-1])
        selected_speed = speed[mask]
        output[stream] = dict(samples=int(mask.sum()),
            cruise_mean_speed_m_s=float(selected_speed.mean()) if mask.any() else None,
            cruise_min_speed_m_s=float(selected_speed.min()) if mask.any() else None,
            cruise_max_speed_m_s=float(selected_speed.max()) if mask.any() else None,
            cruise_speed_within_5pct_of_022_pct=float(100*np.mean(np.abs(selected_speed-.22)<=.011)) if mask.any() else None,
            cruise_rotate_in_place_samples=int(np.sum(mask & (speed<.005) & (np.abs(omega)>.05))),
            all_rotate_in_place_samples=int(np.sum((speed<.005) & (np.abs(omega)>.05))),
            max_abs_omega_rad_s=float(np.abs(omega).max()) if len(omega) else None)
    # Comparable to the pilot scratch's full-run, nearest-vertex arithmetic mean.
    idx = np.argmin(np.linalg.norm(poses[:,None,:2]-path[None,:,:2],axis=2),axis=1)
    yaw_error = np.degrees(np.abs(experiment.wrap(poses[:,2]-path[idx,2])))
    output['pilot_style_heading_mean_deg'] = float(yaw_error.mean())
    output['pilot_style_heading_max_deg'] = float(yaw_error.max())
    return output


def merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict):
            merge(target.setdefault(key, {}), value)
        else:
            target[key] = value


def run_candidate(args, folder, methods, overrides=None, repeats=1):
    import rclpy
    sys.path.insert(0, str(ROOT/'tests'))
    from ros_access_controller_smoke import Plant, controller_stack, spin_for
    from ros_access_end_to_end_smoke import run_while_spinning
    config = yaml.safe_load(args.config.read_text())
    if overrides:
        merge(config['conditions']['E2_environment'].setdefault('controller_overrides', {}), overrides)
    folder.mkdir(parents=True, exist_ok=False)
    config_file = folder/'config.yaml'; config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    session = folder/'session'
    manifest = experiment.prepare(session,args.params,[0.,0.,0.],args.path,config_file=config_file)
    path = np.loadtxt(args.path,delimiter=',',skiprows=1)
    frozen = session/'nav2_params_E2_environment.yaml'
    rclpy.init()
    plant = Plant(bounds=(path[:,:2].min(axis=0)-5, path[:,:2].max(axis=0)+5))
    # Start TF at route start before configuring a local costmap at arbitrary coordinates.
    plant.reset(manifest['starts']['E2_environment']['map_pose'])
    spin_for(plant,.5)
    rows = []
    try:
        with controller_stack(plant,frozen,folder/'stack'):
            snapshots = {}
            for name in ('controller_server','velocity_smoother'):
                snapshots[name] = folder/f'{name}_runtime.yaml'
                snapshots[name].write_text(yaml.safe_dump(experiment.snapshot_parameters(plant,name)))
            for name in methods:
                experiment.verify_runtime_parameters(frozen,snapshots,name)
            for repeat in range(1,repeats+1):
                for name in methods:
                    trial_id = f'E2_environment_{name}_r{repeat}'
                    spin_for(plant,2.3)
                    plant.reset(manifest['starts']['E2_environment']['map_pose'])
                    spin_for(plant,.5)
                    error = None
                    try:
                        run_while_spinning(plant,[sys.executable,str(ROOT/'scripts/dwvp_access_experiment.py'),
                            'run','--session',str(session),'--trial',trial_id],folder/f'{trial_id}.log',
                            config['trial']['timeout_s']+45.)
                    except AssertionError as exc:
                        error = str(exc)
                    summary = experiment.summarize(session)
                    row = next((r for r in summary['trials'] if r['trial_id']==trial_id),
                               dict(trial_id=trial_id,success=False))
                    row = dict(row,controller=name,repeat=repeat,session=str(session),runtime_parameters_verified=True)
                    row['recorder_error'] = error
                    row['timing'] = summary.get('controller_timing',{}).get(trial_id,{})
                    if (session/'runs'/trial_id/'tracking.csv').exists():
                        row['motion'] = motion_diagnostics(session/'runs'/trial_id,path)
                    rows.append(row)
                    experiment.write_json(folder/'results.json',rows)
                    print(json.dumps({k:row.get(k) for k in ('controller','repeat','success',*COMMON_METRICS)}),flush=True)
            spin_for(plant,2.3)
    finally:
        plant.destroy_node(); rclpy.shutdown()
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['sweep','verify'])
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'params/dwvp_access_experiment.yaml')
    parser.add_argument('--params',type=Path,default=ROOT/'params/hsrb_dwvp_access_params.yaml')
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if not Path('/.dockerenv').exists() or set(os.listdir('/sys/class/net')) != {'lo'} or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('Use Docker --network none with ROS_LOCALHOST_ONLY=1')
    if not 1<=args.repeats<=5:
        parser.error('--repeats must be between 1 and 5')
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(selection_rule=RULE,path_sha256=experiment.digest(args.path),
              config_sha256=experiment.digest(args.config),params_sha256=experiment.digest(args.params),
              environment='Synthetic kinematic plant, free map and scans; no physical robot',physical_trials=0,
              grid=[dict(controller=c,candidate=n,overrides=o) for c,n,o in grid()])
    experiment.write_json(args.output/'plan_before_results.json',plan)
    if args.command=='sweep':
        rows=[]
        for controller,name,overrides in grid():
            try:
                trial=run_candidate(args,args.output/name,[controller],{controller:overrides})[0]
            except Exception as exc:
                # A setup/recording failure is a result, not permission to drop a grid entry.
                trial=dict(controller=controller,success=False,setup_error=str(exc))
                print(json.dumps(dict(candidate=name,**trial)),flush=True)
            trial.update(candidate=name,overrides=overrides)
            rows.append(trial)
            experiment.write_json(args.output/'results.json',rows)
        selected=select(rows)
        report=dict(plan=plan,rows=rows,selected=selected)
        experiment.write_json(args.output/'report.json',report)
        with (args.output/'sweep.csv').open('w') as stream:
            fields=['controller','candidate','success','selection_score',*COMMON_METRICS]
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        if any(v is None for v in selected.values()):
            raise SystemExit('No completing candidate for at least one controller; see report.json')
    else:
        rows=run_candidate(args,args.output/'controllers',['RPP','DWPP','MPPI','DWB','DWVP'],repeats=args.repeats)
        report=dict(plan=plan,rows=rows)
        experiment.write_json(args.output/'report.json',report)
        if not all(r['success'] and not r.get('data_errors') and r['timing'].get('failed_calls')==0 for r in rows):
            raise SystemExit('Verification did not pass for every trial; see report.json')


if __name__=='__main__':
    main()
