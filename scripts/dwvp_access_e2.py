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
import time

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


STRAIGHT_RULE = ('Odometry sample mean where projected s is in [0.3,length-0.6] m '
                 'and abs(reference XY tangent derivative) <= 0.1 rad/m. '
                 'Below target flags mean < 0.22 m/s; within-5pct also reported.')


def grid(profile='legacy'):
    result = []
    if profile == 'route-v5':
        settings = yaml.safe_load((ROOT/'params/dwvp_access_e2_route_design_v5.yaml').read_text())['sweep']['DWB']
        for goal in settings['GoalDist.scale']:
            for orientation in settings['PathOrientationCritic.scale']:
                for duration in settings['sim_time']:
                    result.append(('DWB', f'dwb_goal_{goal:g}_yaw_{orientation:g}_time_{duration:g}', {
                        'GoalDist.scale': goal, 'PathOrientationCritic.scale': orientation,
                        'sim_time': duration}))
        return result
    if profile == 'route-v4':
        for duration in (2., 3.):
            for orientation in (4., 16.):
                for position in (16., 64.):
                    result.append(('DWB', f'dwb_time_{duration:g}_yaw_{orientation:g}_position_{position:g}', {
                        'sim_time': duration, 'PathOrientationCritic.scale': orientation,
                        'PathPositionCritic.scale': position}))
        return result
    if profile in ('route-v2', 'route-v3'):
        for angle in (6., 12., 20.):
            for align in (15., 30.):
                for wz in (.2, .4):
                    result.append(('MPPI', f'mppi_angle_{angle:g}_align_{align:g}_wz_{wz:g}', {
                        'PathAngleCritic': {'cost_weight': angle},
                        'PathAlignCritic': {'cost_weight': align, 'use_path_orientations': True},
                        'wz_std': wz}))
        for duration in ((2., 3.) if profile == 'route-v3' else (1.7, 2., 3., 8.)):
            for orientation in ((4., 16., 64.) if profile == 'route-v3' else (1., 4., 16.)):
                for position in ((16., 64.) if profile == 'route-v3' else (4., 16., 64.)):
                    result.append(('DWB', f'dwb_time_{duration:g}_yaw_{orientation:g}_position_{position:g}', {
                        'sim_time': duration, 'PathOrientationCritic.scale': orientation,
                        'PathPositionCritic.scale': position}))
        return sorted(result, key=lambda entry: entry[0] != 'DWB')
    if profile != 'legacy':
        raise ValueError('Unknown sweep profile')
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
    from dwvp_access_path import tangent_path
    arc = experiment.arclength(path[:,:2])
    tangent_curvature = np.gradient(np.unwrap(tangent_path(path[:,:2])[:,2]), arc)
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
        straight = mask & (np.abs(np.interp(progress, arc, tangent_curvature)) <= .1)
        straight_speed = speed[straight]
        output[stream].update(straight_samples=int(straight.sum()),
            straight_cruise_mean_speed_m_s=float(straight_speed.mean()) if straight.any() else None,
            straight_cruise_below_022=bool(straight_speed.mean() < .22) if straight.any() else None,
            straight_cruise_within_5pct_pct=float(100*np.mean(np.abs(straight_speed-.22)<=.011)) if straight.any() else None)
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


def mapped_plant(Plant, map_file):
    """E2-only static map and ideal planar ray sensor; keep the E1 plant unchanged."""
    from dwvp_access_path import load_map
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import LaserScan
    info, pixels, blocked = load_map(map_file)
    occupancy = np.flipud(pixels/255 if info.get('negate', 0) else (255-pixels)/255)
    data = np.full(occupancy.shape, -1, dtype=np.int8)
    data[occupancy < info['free_thresh']] = 0
    data[occupancy > info['occupied_thresh']] = 100
    class MapPlant(Plant):
        def __init__(self):
            super().__init__()
            grid = OccupancyGrid()
            grid.header.frame_id = 'map'
            grid.header.stamp = self.get_clock().now().to_msg()
            grid.info.resolution = float(info['resolution'])
            grid.info.height, grid.info.width = data.shape
            grid.info.origin.position.x, grid.info.origin.position.y = map(float, info['origin'][:2])
            grid.info.origin.orientation.z = math.sin(info['origin'][2]/2)
            grid.info.origin.orientation.w = math.cos(info['origin'][2]/2)
            grid.data = data.ravel().tolist()
            self.map_pub.publish(grid)

        def publish_scan(self):
            scan = LaserScan()
            scan.header.frame_id = 'base_link'
            scan.header.stamp = self.get_clock().now().to_msg()
            scan.angle_min = -math.pi; scan.angle_max = math.pi
            scan.angle_increment = 2*math.pi/180
            scan.range_min = .05; scan.range_max = 5.
            angles = self.yaw+np.linspace(-math.pi, math.pi, 181)-info['origin'][2]
            c, s = math.cos(info['origin'][2]), math.sin(info['origin'][2])
            origin = (np.array([self.x, self.y])-info['origin'][:2])@np.array([[c,-s],[s,c]])
            distances = np.arange(.05, 5., info['resolution']/4)
            endpoints = origin+np.c_[np.cos(angles), np.sin(angles)][:, None, :]*distances[None, :, None]
            cells = np.floor(endpoints/info['resolution']).astype(int)
            x, y = cells[:, :, 0], cells[:, :, 1]
            inside = (x >= 0) & (x < data.shape[1]) & (y >= 0) & (y < data.shape[0])
            hit = np.ones(inside.shape, dtype=bool)
            hit[inside] = blocked[y[inside], x[inside]]
            first = hit.argmax(axis=1)
            scan.ranges = np.where(hit.any(axis=1), distances[first], 5.).astype(float).tolist()
            self.scan_pub.publish(scan)
    return MapPlant()


def reset_costmap(plant, output, path):
    """Remove sensor history after synthetic teleport; retain before/after evidence."""
    from nav2_msgs.srv import ClearEntireCostmap, GetCostmap
    from ros_access_controller_smoke import service, spin_for
    def snapshot(name):
        grid = service(plant, GetCostmap, '/local_costmap/get_costmap', GetCostmap.Request()).map
        meta = grid.metadata
        values = np.asarray(grid.data).reshape(meta.size_y, meta.size_x)
        np.savetxt(output/(name+'.csv'), values, fmt='%d', delimiter=',')
        from dwvp_access_path import inflation_profile
        profile = inflation_profile(path[:50], values,
            [meta.origin.position.x, meta.origin.position.y, 0.], meta.resolution)
        return dict(origin=[meta.origin.position.x, meta.origin.position.y], resolution=meta.resolution,
                    initial_reference_raw_costs=profile['costs'],
                    note='Native raw cost 0..255, first 50 path poses; -2 outside local map')
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot('before_clear')
    service(plant, ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap', ClearEntireCostmap.Request())
    spin_for(plant, .6)
    after = snapshot('after_clear')
    experiment.write_json(output/'reset.json', dict(before=before, after=after))


class DwbDiagnostics:
    """Save critic rejection counts and best scores; diagnostic runs are unscored."""
    def __init__(self, plant, folder, delay=30.):
        from dwb_msgs.msg import LocalPlanEvaluation
        self.file = (folder/'dwb_evaluation.jsonl').open('w')
        self.last = 0.
        self.plant = plant
        self.subscription = None
        self.started = time.monotonic()
        # Full native evaluations contain thousands of trajectories. Capture a
        # short window only, avoiding instrumentation during the approach.
        def window():
            elapsed = time.monotonic()-self.started
            if elapsed >= delay+5.:
                if self.subscription:
                    plant.destroy_subscription(self.subscription)
                    self.subscription = None
                self.timer.cancel()
            elif elapsed >= delay and self.subscription is None:
                self.subscription = plant.create_subscription(LocalPlanEvaluation, '/evaluation', self.receive, 1)
        self.timer = plant.create_timer(.25, window)

    def receive(self, message):
        stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        if stamp-self.last < .5 or not message.twists:
            return
        self.last = stamp
        rejected = {}
        for trajectory in message.twists:
            if trajectory.total < 0:
                for score in trajectory.scores:
                    if score.raw_score < 0:
                        rejected[score.name] = rejected.get(score.name, 0)+1
        best = message.twists[message.best_index]
        velocity = best.traj.velocity
        row = dict(stamp_s=stamp, pose=[self.plant.x, self.plant.y, self.plant.yaw],
                   candidates=len(message.twists), rejected_by_critic=rejected,
                   best_velocity=[velocity.x, velocity.y, velocity.theta], best_total=best.total,
                   best_scores={s.name:dict(raw=s.raw_score, scale=s.scale) for s in best.scores})
        row['straight_candidates'] = [dict(vx=t.traj.velocity.x, total=t.total,
            scores={s.name:s.raw_score for s in t.scores}) for t in message.twists
            if abs(t.traj.velocity.y) < 1e-9 and abs(t.traj.velocity.theta) < 1e-9]
        row['critic_candidate_ranges'] = {}
        for name in {s.name for t in message.twists for s in t.scores}:
            values = [s.raw_score for t in message.twists for s in t.scores
                      if s.name == name and s.raw_score >= 0]
            if values:
                row['critic_candidate_ranges'][name] = dict(count=len(values),
                    minimum=min(values), maximum=max(values), spread=max(values)-min(values),
                    unique=len(set(values)))
        self.file.write(json.dumps(row)+'\n'); self.file.flush()

    def close(self):
        if self.subscription:
            self.plant.destroy_subscription(self.subscription)
        self.plant.destroy_timer(self.timer)
        self.file.close()


class CostmapObservations:
    """Record native costmap update stamps at the configured publication rate.

    This observes publication and age, not every internal update cycle. Missed
    update-loop warnings remain in controller_server.log.
    """
    def __init__(self, plant, folder):
        from nav2_msgs.msg import Costmap
        self.plant = plant
        self.file = (folder/'costmap_observations.csv').open('w')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['receive_stamp_s', 'update_stamp_s', 'age_s', 'resolution_m', 'width', 'height'])
        self.subscription = plant.create_subscription(Costmap, '/local_costmap/costmap_raw', self.receive, 1)

    def receive(self, message):
        stamp = message.metadata.update_time.sec+message.metadata.update_time.nanosec*1e-9
        now = self.plant.get_clock().now().nanoseconds*1e-9
        self.writer.writerow([now, stamp, now-stamp, message.metadata.resolution,
                              message.metadata.size_x, message.metadata.size_y])
        self.file.flush()

    def close(self):
        self.plant.destroy_subscription(self.subscription)
        self.file.close()


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
    manifest = experiment.prepare(session,args.params,[0.,0.,0.],args.path,config_file=config_file,
                                  conditions=['E2_environment'],repeats=list(range(1,repeats+1)))
    path = np.loadtxt(args.path,delimiter=',',skiprows=1)
    frozen = session/'nav2_params_E2_environment.yaml'
    rclpy.init()
    plant = (mapped_plant(Plant, args.map) if getattr(args, 'map', None) else
             Plant(bounds=(path[:,:2].min(axis=0)-5, path[:,:2].max(axis=0)+5)))
    diagnostics = DwbDiagnostics(plant, folder, args.diagnostic_after) if getattr(args, 'diagnose_dwb', False) else None
    costmap_observations = CostmapObservations(plant, folder)
    # Start TF at route start before configuring a local costmap at arbitrary coordinates.
    plant.reset(manifest['starts']['E2_environment']['map_pose'])
    spin_for(plant,.5)
    rows = []
    try:
        with controller_stack(plant,frozen,folder/'stack'):
            snapshots = {}
            for name in ('controller_server','velocity_smoother','local_costmap/local_costmap'):
                snapshots[name] = folder/f'{name}_runtime.yaml'
                snapshots[name].parent.mkdir(parents=True, exist_ok=True)
                snapshots[name].write_text(yaml.safe_dump(experiment.snapshot_parameters(plant,name)))
            for name in methods:
                experiment.verify_runtime_parameters(frozen,snapshots,name)
            inflation = config['conditions']['E2_environment'].get('costmap_inflation_radius_m')
            if inflation is not None:
                snapshot = yaml.safe_load(snapshots['local_costmap/local_costmap'].read_text())
                actual = snapshot['/local_costmap/local_costmap']['ros__parameters']
                if actual['inflation_layer.inflation_radius'] != inflation or actual['robot_radius'] != .22:
                    raise ValueError('E2 local costmap runtime radius does not match the frozen protocol')
                resolution = config['conditions']['E2_environment'].get('local_costmap_resolution_m')
                if resolution is not None and not math.isclose(actual['resolution'], resolution):
                    raise ValueError('E2 local costmap runtime resolution does not match the frozen protocol')
            for repeat in range(1,repeats+1):
                for name in methods:
                    trial_id = f'E2_environment_{name}_r{repeat}'
                    spin_for(plant,2.3)
                    plant.reset(manifest['starts']['E2_environment']['map_pose'])
                    spin_for(plant,.5)
                    if getattr(args, 'clear_costmap_on_reset', False):
                        reset_costmap(plant, folder/'costmap_resets'/trial_id, path)
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
                        odom = row['motion'].get('odom', {})
                        row['straight_cruise_mean_speed_m_s'] = odom.get('straight_cruise_mean_speed_m_s')
                        row['straight_cruise_below_022'] = odom.get('straight_cruise_below_022')
                    rows.append(row)
                    experiment.write_json(folder/'results.json',rows)
                    print(json.dumps({k:row.get(k) for k in ('controller','repeat','success',*COMMON_METRICS)}),flush=True)
            spin_for(plant,2.3)
    finally:
        costmap_observations.close()
        if diagnostics:
            diagnostics.close()
        plant.destroy_node(); rclpy.shutdown()
    return rows


def run_grid_candidate(args, index, entry):
    controller, name, overrides = entry
    # Independent synthetic ROS graphs within the network-none container.
    os.environ['ROS_DOMAIN_ID'] = str(180 + index)
    os.environ['ROS_HOME'] = f'/tmp/ros-e2-{index}'
    try:
        trial = run_candidate(args, args.output/name, [controller], {controller: overrides})[0]
    except Exception as exc:
        trial = dict(controller=controller, success=False, setup_error=str(exc))
        print(json.dumps(dict(candidate=name, **trial)), flush=True)
    trial.update(candidate=name, overrides=overrides)
    return index, trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['sweep','verify'])
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'params/dwvp_access_experiment.yaml')
    parser.add_argument('--params',type=Path,default=ROOT/'params/hsrb_dwvp_access_params.yaml')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--map', type=Path, help='E2 static occupancy map and synthetic ray scans')
    parser.add_argument('--diagnose-dwb', action='store_true', help='Record DWB evaluations; affects timing, use separately')
    parser.add_argument('--diagnostic-after', type=float, default=30., help='Begin a 5 s evaluation capture after this delay')
    parser.add_argument('--clear-costmap-on-reset', action='store_true', help='Clear synthetic scan history after teleport and save costmap snapshots')
    parser.add_argument('--grid-profile', choices=['legacy', 'route-v2', 'route-v3', 'route-v4', 'route-v5'], default='legacy')
    parser.add_argument('--sweep-workers', type=int, default=1,
                        help='Isolated ROS processes for sweep only; verification stays sequential')
    parser.add_argument('--dwvp-fixed-lookahead',type=float,
                        help='Also verify DWVP with this fixed lookahead, in a separate frozen session')
    args=parser.parse_args()
    if not Path('/.dockerenv').exists() or set(os.listdir('/sys/class/net')) != {'lo'} or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('Use Docker --network none with ROS_LOCALHOST_ONLY=1')
    if not 1<=args.repeats<=5:
        parser.error('--repeats must be between 1 and 5')
    if not 1 <= args.sweep_workers <= 4:
        parser.error('--sweep-workers must be between 1 and 4')
    if args.dwvp_fixed_lookahead is not None and (not math.isfinite(args.dwvp_fixed_lookahead) or args.dwvp_fixed_lookahead <= 0):
        parser.error('--dwvp-fixed-lookahead must be positive and finite')
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(selection_rule=RULE,path_sha256=experiment.digest(args.path),
              config_sha256=experiment.digest(args.config),params_sha256=experiment.digest(args.params),
              environment=('Synthetic kinematic plant, supplied static map and ideal ray scans' if args.map else
                           'Synthetic kinematic plant, free map and scans; no physical robot'),physical_trials=0,
              map_yaml_sha256=experiment.digest(args.map) if args.map else None,
              diagnostic_evaluation_recording=args.diagnose_dwb,
              diagnostic_capture_after_s=args.diagnostic_after if args.diagnose_dwb else None,
              clear_costmap_on_reset=args.clear_costmap_on_reset,
              grid_profile=args.grid_profile, straight_cruise_rule=STRAIGHT_RULE,
              sweep_workers=args.sweep_workers if args.command == 'sweep' else 1,
              grid=[dict(controller=c,candidate=n,overrides=o) for c,n,o in grid(args.grid_profile)],
              repeats=1 if args.command=='sweep' else args.repeats,
              dwvp_fixed_lookahead_m=args.dwvp_fixed_lookahead)
    experiment.write_json(args.output/'plan_before_results.json',plan)
    if args.command=='sweep':
        entries = grid(args.grid_profile)
        if args.sweep_workers == 1:
            rows=[]
            for index, entry in enumerate(entries):
                _, trial = run_grid_candidate(args, index, entry)
                rows.append(trial)
                experiment.write_json(args.output/'results.json',rows)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            completed = {}
            with ProcessPoolExecutor(max_workers=args.sweep_workers,
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(run_grid_candidate, args, i, entry) for i, entry in enumerate(entries)]
                for future in as_completed(futures):
                    index, trial = future.result()
                    completed[index] = trial
                    rows = [completed[i] for i in sorted(completed)]
                    experiment.write_json(args.output/'results.json', rows)
        selected=select(rows)
        report=dict(plan=plan,rows=rows,selected=selected)
        experiment.write_json(args.output/'report.json',report)
        with (args.output/'sweep.csv').open('w') as stream:
            fields=['controller','candidate','success','selection_score',*COMMON_METRICS,
                    'straight_cruise_mean_speed_m_s','straight_cruise_below_022']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        if any(selected[c] is None for c in {entry[0] for entry in entries}):
            raise SystemExit('No completing candidate for at least one controller; see report.json')
    else:
        methods=yaml.safe_load(args.config.read_text())['conditions']['E2_environment']['methods']
        rows=run_candidate(args,args.output/'controllers',methods,repeats=args.repeats)
        for row in rows:
            row['variant']=row['controller']
        experiment.write_json(args.output/'report.json',dict(plan=plan,rows=rows))
        if args.dwvp_fixed_lookahead is not None:
            if 'DWVP' not in methods:
                raise ValueError('The fixed-lookahead variant requires DWVP in E2 methods')
            fixed=run_candidate(args,args.output/'dwvp_fixed',['DWVP'],
                {'DWVP':{'use_velocity_scaled_lookahead_dist':False,
                         'lookahead_dist':args.dwvp_fixed_lookahead,
                         'min_lookahead_dist':args.dwvp_fixed_lookahead,
                         'max_lookahead_dist':args.dwvp_fixed_lookahead}},repeats=args.repeats)
            for row in fixed:
                row['variant']='DWVP_fixed'
            rows.extend(fixed)
        report=dict(plan=plan,rows=rows)
        experiment.write_json(args.output/'report.json',report)
        if not all(r['success'] and not r.get('data_errors') and r['timing'].get('failed_calls')==0 for r in rows):
            raise SystemExit('Verification did not pass for every trial; see report.json')


if __name__=='__main__':
    main()
