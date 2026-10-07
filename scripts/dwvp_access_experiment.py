#!/usr/bin/env python3
"""Prepare 80 fixed-path trials, record one ROS trial, and summarize completed data.

Offline preparation and analysis do not import ROS. ``prepare --start-from-current``
observes localization without motion. ``run`` sends one FollowPath goal; the batch
tool also uses this recorder for separately stored relocations. Origins are frozen
across methods; measured-start sessions use a separate origin for each E1 condition.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


# Independently synchronized robot/PC clocks can differ slightly. Allow at most
# 50 ms of future source time (11 mm per axis at the 0.22 m/s axis limit).
# The real HSR produced source stamps about 24 ms ahead of the PC after NTP
# synchronization; a 20 ms bound repeatedly reset stationary capture.
# Receive ages and command ordering use the local clock and remain nonnegative.
# Keep signed ages in recordings; never clamp or rewrite source timestamps.
CLOCK_FUTURE_TOLERANCE_S = .05
LEGACY_CLOCK_FUTURE_TOLERANCE_S = .02
TURNAROUND_GOAL_CHECKER = 'turnaround_goal_checker'
TURNAROUND_YAW_TOLERANCE = .005
ALIGNMENT_GOAL_CHECKER = 'alignment_goal_checker'


def with_alignment_checker(parameters):
    """Aim inside the frozen start tolerance during unscored positioning."""
    parameters = copy.deepcopy(parameters)
    server = parameters['controller_server']['ros__parameters']
    if ALIGNMENT_GOAL_CHECKER in server['goal_checker_plugins']:
        raise ValueError('Alignment checker must not be part of frozen trial parameters')
    general = server['general_goal_checker']
    server['goal_checker_plugins'] = [*server['goal_checker_plugins'], ALIGNMENT_GOAL_CHECKER]
    server[ALIGNMENT_GOAL_CHECKER] = dict(plugin='nav2_controller::SimpleGoalChecker',
        stateful=False, xy_goal_tolerance=min(.05, general['xy_goal_tolerance']/2),
        yaw_goal_tolerance=min(.15, general['yaw_goal_tolerance']/2))
    return parameters


def with_turnaround_checker(parameters):
    """Add an unscored in-place heading checker without changing trial tuning."""
    parameters = copy.deepcopy(parameters)
    server = parameters['controller_server']['ros__parameters']
    if TURNAROUND_GOAL_CHECKER in server['goal_checker_plugins']:
        raise ValueError('Turnaround checker must not be part of frozen trial parameters')
    server['goal_checker_plugins'] = [*server['goal_checker_plugins'], TURNAROUND_GOAL_CHECKER]
    server[TURNAROUND_GOAL_CHECKER] = dict(plugin='nav2_controller::SimpleGoalChecker',
        stateful=False, xy_goal_tolerance=server['general_goal_checker']['xy_goal_tolerance'],
        yaw_goal_tolerance=TURNAROUND_YAW_TOLERANCE)
    return parameters


def transfer_goal_checker(transfer, reference):
    checker = transfer.get('goal_checker_id', 'general_goal_checker')
    if checker == TURNAROUND_GOAL_CHECKER:
        if (transfer.get('endpoint_policy') != 'reanchor_after_stop'
                or reference.shape != (2, 3) or not np.isfinite(reference).all()
                or np.linalg.norm(reference[1, :2]-reference[0, :2]) > 1e-6):
            raise ValueError('Turnaround checker is only for unscored in-place turns')
    elif checker == ALIGNMENT_GOAL_CHECKER:
        if transfer.get('endpoint_policy', 'fixed_tolerance') != 'fixed_tolerance':
            raise ValueError('Alignment checker requires a fixed target')
    elif checker != 'general_goal_checker':
        raise ValueError('Unknown transfer goal checker')
    return checker


def source_age_is_fresh(age, maximum_age=.2, *, future_tolerance=CLOCK_FUTURE_TOLERANCE_S):
    """Scalar/array freshness for remote TF/sensor stamps, not receive times."""
    return (age >= -future_tolerance) & (age <= maximum_age)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def arclength(xy):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


CONTROLLERS = ('RPP', 'DWPP', 'MPPI', 'DWB', 'VP_CLIP', 'VP_SCALED', 'DWVP')
CONDITIONS = ('E1_lateral', 'E1_orientation_nominal', 'E1_orientation_half',
              'E1_orientation_quarter', 'E2_environment')


def default_config():
    import yaml
    local = Path(__file__).resolve().parents[1] / 'params/dwvp_access_experiment.yaml'
    if not local.exists():
        from ament_index_python.packages import get_package_share_directory
        local = Path(get_package_share_directory('dwpp_test_simulation')) / 'params/dwvp_access_experiment.yaml'
    return yaml.safe_load(local.read_text())


def condition_common(config, condition):
    """One source for controller, smoother and metric acceleration/deceleration."""
    common = copy.deepcopy(config['common'])
    scale = config['conditions'][condition]['acceleration_scale']
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Condition acceleration_scale must be positive and finite')
    for key in ('max_accel', 'max_decel'):
        common[key] = [float(value * scale) for value in common[key]]
    return common


def render_parameters(params, config, condition=None):
    import yaml
    values = yaml.safe_load(Path(params).read_text())
    c = condition_common(config, condition) if condition else config['common']
    cs = values['controller_server']['ros__parameters']
    cs['controller_frequency'] = float(c['control_frequency_hz'])
    cs['general_goal_checker'].update(xy_goal_tolerance=config['trial']['xy_tolerance_m'],
                                      yaw_goal_tolerance=config['trial']['yaw_tolerance_rad'])
    hi, lo, accel, decel = (c[k] for k in ('max_velocity', 'min_velocity', 'max_accel', 'max_decel'))
    if not (len(hi) == len(lo) == len(accel) == len(decel) == 3):
        raise ValueError('Velocity and acceleration limits require three components')
    if not np.isfinite([*hi, *lo, *accel, *decel]).all() or any(v <= 0 for v in hi + accel) or any(v >= 0 for v in lo + decel):
        raise ValueError('Finite positive upper/acceleration and negative lower/deceleration limits required')
    if hi != [-x for x in lo] or accel != [-x for x in decel] or hi[0] != hi[1] or accel[0] != accel[1]:
        raise ValueError('This seven-controller comparison requires symmetric equal planar limits')
    if c['control_frequency_hz'] <= 0 or not (0 < c['min_lookahead_dist'] <= c['max_lookahead_dist']) or c['lookahead_time'] <= 0:
        raise ValueError('Invalid frequency or lookahead settings')
    for name in CONTROLLERS:
        ctrl = cs[name]
        if name not in ('MPPI', 'DWB'):
            ctrl.update({key: float(c[key]) for key in ('lookahead_time', 'min_lookahead_dist', 'max_lookahead_dist')})
            ctrl['lookahead_dist'] = float(c['min_lookahead_dist'])
            ctrl['max_angular_accel'] = accel[2]
        if name in ('RPP', 'DWPP'):
            ctrl['desired_linear_vel'] = c['pp_translation_speed']
            ctrl['rotate_to_heading_angular_vel'] = hi[2]
        if name == 'DWPP':
            ctrl.update(max_linear_vel=hi[0], min_linear_vel=0.0, max_angular_vel=hi[2], min_angular_vel=lo[2],
                        max_linear_accel=accel[0], max_linear_decel=decel[0], max_angular_decel=decel[2])
        if name in ('DWVP', 'VP_CLIP', 'VP_SCALED'):
            ctrl.update(desired_linear_vel=c['omni_regulation_speed'], vp_translation_speed=c['vp_translation_speed'])
            for i, axis in enumerate(('x', 'y', 'theta')):
                ctrl.update({f'max_vel_{axis}': hi[i], f'min_vel_{axis}': lo[i], f'max_accel_{axis}': accel[i]})
        if name == 'DWB':
            ctrl.update(min_vel_x=lo[0], max_vel_x=hi[0], min_vel_y=lo[1], max_vel_y=hi[1],
                        max_vel_theta=hi[2], max_speed_xy=float(math.hypot(hi[0], hi[1])),
                        sim_period=1.0/c['control_frequency_hz'],
                        xy_goal_tolerance=config['trial']['xy_tolerance_m'])
            for i, axis in enumerate(('x', 'y', 'theta')):
                ctrl.update({f'acc_lim_{axis}': accel[i], f'decel_lim_{axis}': decel[i]})
            if ctrl.get('trajectory_generator_name') == 'dwb_plugins::StandardTrajectoryGenerator':
                # Only LimitedAccelGenerator declares sim_period. The standard
                # rollout advances acceleration at its configured time steps.
                ctrl.pop('sim_period', None)
                if ctrl.get('discretize_by_time') and ctrl.get('limit_vel_cmd_in_traj'):
                    ctrl['time_granularity'] = 1.0/c['control_frequency_hz']
        if name == 'MPPI':
            ctrl.update(vx_max=hi[0], vx_min=lo[0], vy_max=hi[1], wz_max=hi[2], model_dt=1.0/c['control_frequency_hz'])
    smoother = values['velocity_smoother']['ros__parameters']
    smoother.update({key: c[key] for key in ('max_velocity', 'min_velocity', 'max_accel', 'max_decel')})
    smoother['smoothing_frequency'] = float(c['control_frequency_hz'])
    return values


def canonical_path(name, config=None):
    settings = (config or default_config())['conditions'][name]
    spacing = settings['spacing_m']
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Path spacing must be positive and finite')
    def samples(length):
        if not np.isfinite(length) or length <= 0:
            raise ValueError('Path dimensions must be positive and finite')
        return np.linspace(0, length, max(2, int(math.ceil(length / spacing)) + 1))
    if name == 'E1_lateral':
        x = samples(settings['length_m'])
        return np.c_[x, np.zeros(len(x)), np.zeros(len(x))]
    if name in ('E1_orientation_nominal', 'E1_orientation_half', 'E1_orientation_quarter'):
        x = samples(settings['length_m'])
        start, length = settings['orientation_start_m'], settings['orientation_length_m']
        if not (0 < start < start + length < settings['length_m']):
            raise ValueError('Orientation transition must lie inside the path')
        yaw = np.clip((x - start) / length, 0., 1.) * settings['goal_yaw_rad']
        return np.c_[x, np.zeros(len(x)), yaw]
    raise ValueError(name)


def transform_poses(poses, origin):
    poses = np.asarray(poses, dtype=float).copy()
    ox, oy, angle = origin
    c, s = math.cos(angle), math.sin(angle)
    poses[..., :2] = poses[..., :2] @ np.array([[c, s], [-s, c]]) + [ox, oy]
    poses[..., 2] = wrap(poses[..., 2] + angle)
    return poses


def start_pose(manifest, trial):
    return np.asarray(trial.get('start_pose', manifest['starts'][trial['task']]['map_pose']), dtype=float)


def origin_from_start(current, local_start):
    """Invert the rigid transform so a condition starts at the measured pose."""
    current, local_start = np.asarray(current, dtype=float), np.asarray(local_start, dtype=float)
    if current.shape != (3,) or local_start.shape != (3,) or not np.isfinite([current, local_start]).all():
        raise ValueError('Finite x,y,yaw start poses required')
    yaw = float(wrap(current[2] - local_start[2]))
    offset = transform_poses(local_start, [0., 0., yaw])
    return [float(current[0] - offset[0]), float(current[1] - offset[1]), yaw]


def reanchor_trial(manifest, trial, reference, measured_pose):
    """Rigidly place an E1 trial at its measured robot start, retaining offsets."""
    if trial['task'] == 'E2_environment':
        raise ValueError('Per-trial current starts support E1 only; E2 routes stay map-fixed')
    transform = origin_from_start(measured_pose, start_pose(manifest, trial))
    placed = dict(trial, start_pose=list(map(float, measured_pose)))
    return placed, validate_path(transform_poses(reference, transform))


def recorded_geometry(session, trial, folder):
    """Load actual recorded geometry, validating a per-trial placement if present."""
    folder = Path(folder)
    manifest, frozen, template = load_trial(session, trial['id'])
    path = validate_path(np.atleast_2d(np.loadtxt(folder/'reference.csv', delimiter=',', skiprows=1)))
    metadata_file = folder/'trial.json'
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    policy = metadata.get('reference_policy', 'session_fixed')
    if policy == 'per_trial_current_pose':
        if (metadata.get('trial') != frozen or metadata.get('manifest_sha256') != digest(Path(session)/'manifest.json')
                or metadata.get('reference_sha256') != digest(folder/'reference.csv')
                or metadata.get('start_capture_sha256') != digest(folder/'start_capture.json')):
            raise ValueError('Recorded current-start geometry or capture changed')
        captured = json.loads((folder/'start_capture.json').read_text())['capture']['pose']
        placed, expected = reanchor_trial(manifest, frozen, template, captured)
        if path.shape != expected.shape or not np.allclose(path, expected, atol=1e-9, rtol=0):
            raise ValueError('Recorded reference differs from the captured E1 placement')
        return start_pose(manifest, placed), path, policy
    if policy != 'session_fixed':
        raise ValueError('Unknown recorded reference policy')
    return start_pose(manifest, frozen), path, policy


def check_start_pose(current, manifest, trial):
    wanted = start_pose(manifest, trial)
    if np.linalg.norm(np.asarray(current[:2]) - wanted[:2]) > manifest['xy_tolerance_m'] or abs(float(wrap(current[2]-wanted[2]))) > manifest['yaw_tolerance_rad']:
        raise RuntimeError('Reset robot to the frozen condition start pose before this trial')


def validate_path(path):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('A path must have at least two finite x,y,yaw rows')
    if np.any(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) <= 1e-9):
        raise ValueError('Duplicate position samples are not supported')
    return path


def prepare(output, params, origin=None, environment_path=None, seed=20261004, config_file=None,
            *, current_start=None, bidirectional=False, conditions=None, repeats=None):
    import yaml
    config = yaml.safe_load(Path(config_file).read_text()) if config_file else default_config()
    if current_start is not None:
        if origin is not None:
            raise ValueError('Choose --origin or a measured current start, not both')
        # E1 conditions share the measured robot start, not their path origins.
        origins = {name: origin_from_start(current_start, config['conditions'][name]['start_pose'])
                   for name in CONDITIONS if name != 'E2_environment'}
        origin = origins['E1_lateral']
    else:
        origins = {}
    selected_conditions = [c for item in (conditions or CONDITIONS) for c in
                           (CONDITIONS[:-1] if item == 'E1' else [item])]
    selected_repeats = list(range(1, 6)) if repeats is None else sorted(set(repeats))
    if not selected_conditions or not set(selected_conditions) <= set(CONDITIONS):
        raise ValueError('Unknown or empty condition selection')
    if not selected_repeats or not set(selected_repeats) <= set(range(1, 6)):
        raise ValueError('Repeats must be in 1..5')
    if bidirectional and (origin is None or ('E2_environment' in selected_conditions and environment_path is None)):
        raise ValueError('Bidirectional preparation requires a start/origin and an E2 route when E2 is selected')
    # Reject non-finite settings before reserving the output directory.
    def finite_settings(value):
        if isinstance(value, dict):
            for item in value.values(): finite_settings(item)
        elif isinstance(value, list):
            for item in value: finite_settings(item)
        elif isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError('Configuration numbers must be finite')
    finite_settings(config)
    if any(config['trial'][k] <= 0 for k in ('xy_tolerance_m','yaw_tolerance_rad','timeout_s')):
        raise ValueError('Trial tolerances and timeout must be positive')
    if config['metrics']['lateral_deadband_m'] < 0 or not 0 < config['metrics']['lateral_convergence_ratio'] < 1:
        raise ValueError('Invalid lateral metric settings')
    if config['metrics']['maximum_source_age_s'] <= 0 or config['metrics']['robot_radius_m'] <= 0 or config['metrics']['obstacle_near_range_m'] <= 0:
        raise ValueError('Metric age, footprint radius and near range must be positive')
    if not 0 <= config['metrics']['maximum_unknown_command_pct'] <= 100:
        raise ValueError('Unknown command threshold must be between 0 and 100 percent')
    for name in CONDITIONS:
        window = config['conditions'][name]['evaluation']
        if window['start_m'] < 0 or window['goal_margin_m'] < 0:
            raise ValueError('Evaluation start and goal margin must be nonnegative')
        methods = config['conditions'][name]['methods']
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(CONTROLLERS):
            raise ValueError('Condition methods must be a nonempty unique selection of controllers')
        condition_common(config, name)
        pose = config['conditions'][name]['start_pose']
        if pose is not None and (len(pose) != 3 or not np.isfinite(pose).all()):
            raise ValueError('Condition start pose must be finite x,y,yaw')
    for obstacle in config['surveyed_obstacles']:
        if obstacle['type'] == 'circle':
            if len(obstacle['center']) != 2 or obstacle['radius'] <= 0:
                raise ValueError('Survey circle requires centre and positive radius')
        elif obstacle['type'] == 'polygon':
            vertices = np.asarray(obstacle['vertices'])
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
                raise ValueError('Survey polygon requires at least three 2D vertices')
        else:
            raise ValueError('Survey geometry supports circle and polygon')
    rendered = render_parameters(params, config)
    if origin is not None and (len(origin) != 3 or not np.isfinite(origin).all()):
        raise ValueError('Origin must be a finite x,y,yaw pose')
    routes = {name: validate_path(canonical_path(name, config)) for name in CONDITIONS if name != 'E2_environment'}
    if environment_path is not None:
        routes['E2_environment'] = validate_path(np.loadtxt(environment_path, delimiter=',', skiprows=1))
    for name, route in routes.items():
        window = config['conditions'][name]['evaluation']
        if arclength(route[:, :2])[-1] <= window['start_m'] + window['goal_margin_m']:
            raise ValueError('Path must be longer than evaluation start plus goal margin')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'paths').mkdir()
    (output / 'nav2_params.yaml').write_text(yaml.safe_dump(rendered, sort_keys=False))
    (output / 'experiment_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(params, output / 'params_template.yaml')
    parameter_sets = {}
    for name in CONDITIONS:
        filename = f'nav2_params_{name}.yaml'
        (output / filename).write_text(yaml.safe_dump(render_parameters(params, config, name), sort_keys=False))
        parameter_sets[name] = {'file': filename, 'sha256': digest(output / filename)}
    paths, starts = {}, {}
    for name in CONDITIONS:
        if name not in routes:
            paths[name] = {'file': None, 'status': 'needs_navfn_environment_route', 'frame': 'map'}
            starts[name] = {'map_pose': None}
            continue
        path = routes[name]
        file = output / 'paths' / (name + '.csv')
        np.savetxt(file, path, delimiter=',', header='x,y,yaw', comments='')
        paths[name] = {'file': str(file.relative_to(output)), 'sha256': digest(file),
                       'frame': 'map' if name == 'E2_environment' else 'local'}
        pose = config['conditions'][name]['start_pose']
        if name == 'E2_environment':
            metadata = Path(environment_path).parent / 'path_metadata.json'
            if metadata.exists():
                info = json.loads(metadata.read_text())
                if info['csv_sha256'] != digest(environment_path):
                    raise ValueError('Environment CSV differs from planner metadata')
                shutil.copy2(metadata, output / 'environment_path_metadata.json')
                if pose is None:
                    pose = info['start']
            if pose is None:
                pose = path[0].tolist()
            starts[name] = {'map_pose': pose, 'source_frame': 'map'}
        else:
            if len(pose) != 3 or not np.isfinite(pose).all():
                raise ValueError('Condition start pose must be finite x,y,yaw')
            starts[name] = {'local_pose': pose, 'map_pose': transform_poses(pose, origins.get(name, origin)).tolist() if origin is not None else None}
    trials = []
    rng = random.Random(seed)
    for repeat in selected_repeats:
        for task in CONDITIONS:
            if task not in selected_conditions:
                continue
            methods = list(config['conditions'][task]['methods'])
            rng.shuffle(methods)
            trials.extend({'id': f'{task}_{controller}_r{repeat}', 'task': task,
                           'controller': controller, 'repeat': repeat,
                           'acceleration_scale': config['conditions'][task]['acceleration_scale'],
                           'params_file': parameter_sets[task]['file'],
                           'params_sha256': parameter_sets[task]['sha256']} for controller in methods)
    if bidirectional:
        for index, trial in enumerate(trials):
            task = trial['task']
            trial['direction'] = 'forward' if index % 2 == 0 else 'reverse'
            trial['start_pose'] = starts[task]['map_pose']
            if task == 'E2_environment':
                if trial['direction'] == 'reverse':
                    end = routes[task][-1].copy()
                    end[2] = wrap(end[2] + np.pi)
                    trial['start_pose'] = end.tolist()
            else:
                forward_origin = origins.get(task, origin)
                trial['map_origin'] = forward_origin
                if trial['direction'] == 'reverse':
                    end = transform_poses(routes[task][-1], forward_origin)
                    end[2] = wrap(starts[task]['map_pose'][2] + np.pi)
                    trial['start_pose'] = end.tolist()
                    trial['map_origin'] = origin_from_start(end, config['conditions'][task]['start_pose'])
    manifest = {'schema_version': 3, 'seed': seed, 'frame': 'map', 'map_origin': origin,
                'params_file': 'nav2_params.yaml', 'params_sha256': digest(output / 'nav2_params.yaml'),
                'config_file': 'experiment_config.yaml', 'config_sha256': digest(output / 'experiment_config.yaml'),
                'control_frequency_hz': config['common']['control_frequency_hz'], **config['trial'],
                'paths': paths, 'starts': starts, 'trials': trials, 'parameter_sets': parameter_sets,
                'software_manifest_status': 'Regenerate after commit; working tree is not a committed release',
                'status': 'prepared_unrecorded'}
    if bidirectional:
        manifest['bidirectional'] = True
        manifest['direction_policy'] = 'alternate scored forward/reverse legs; rotate canonical E1 geometry by pi for reverse'
    if current_start is not None:
        manifest['map_origins'] = origins
        manifest['measured_start'] = list(map(float, current_start))
        manifest['origin_policy'] = 'E1 condition starts share one measured map pose; E2 remains map-fixed'
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_trial(session, trial_id):
    session = Path(session)
    manifest = json.loads((session / 'manifest.json').read_text())
    if manifest.get('schema_version') != 3:
        raise ValueError('Unsupported session schema; prepare a new session')
    trial = next((t for t in manifest['trials'] if t['id'] == trial_id), None)
    if trial is None:
        raise ValueError('Unknown or unassigned condition/controller trial ID')
    if manifest['map_origin'] is None:
        raise ValueError('Prepare a session with a surveyed --origin X Y YAW before recording')
    record = manifest['paths'][trial['task']]
    if record.get('file') is None:
        raise ValueError('The NavFn environment route has not been supplied')
    frozen = [(record['file'], record['sha256']), (trial['params_file'], trial['params_sha256']),
              (manifest['config_file'], manifest['config_sha256'])]
    if any(digest(session / file) != sha for file, sha in frozen):
        raise ValueError('Frozen path/configuration changed; prepare a new session')
    import yaml
    config = yaml.safe_load((session / manifest['config_file']).read_text())
    if trial['controller'] not in config['conditions'][trial['task']]['methods']:
        raise ValueError('Unassigned condition/controller combination')
    expected = manifest['parameter_sets'][trial['task']]
    if (trial['params_file'], trial['params_sha256']) != (expected['file'], expected['sha256']):
        raise ValueError('Trial does not use the frozen condition parameters')
    if trial['acceleration_scale'] != config['conditions'][trial['task']]['acceleration_scale']:
        raise ValueError('Trial acceleration profile differs from the frozen configuration')
    path = validate_path(np.loadtxt(session / record['file'], delimiter=',', skiprows=1))
    if record['frame'] == 'local':
        path = transform_poses(path, trial.get('map_origin', manifest.get('map_origins', {}).get(trial['task'], manifest['map_origin'])))
    elif trial.get('direction') == 'reverse':
        path = path[::-1].copy()
        path[:, 2] = wrap(path[:, 2] + np.pi)
    return manifest, trial, path


def tracking_errors(poses, path, signed=False):
    """Project position onto segments and use that same location for yaw error."""
    from dwvp_access_metrics import project_tracking
    errors = project_tracking(poses, path)[:, :2]
    if not signed:
        errors[:, 1] = np.abs(errors[:, 1])
    return errors


def flatten_parameters(values, prefix=''):
    result = {}
    for key, value in values.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten_parameters(value, name + '.'))
        else:
            result[name] = value
    return result


def verify_runtime_parameters(expected_file, runtime_files, controller):
    """Check the experimental contract against parameters read from running nodes."""
    import yaml
    expected = yaml.safe_load(Path(expected_file).read_text())
    for node in ('controller_server', 'velocity_smoother'):
        snapshot = yaml.safe_load(Path(runtime_files[node]).read_text())
        actual = snapshot.get('/' + node, snapshot.get(node))
        if actual is None:
            raise ValueError('Runtime snapshot is missing node ' + node)
        actual = flatten_parameters(actual['ros__parameters'])
        wanted = flatten_parameters(expected[node]['ros__parameters'])
        # Freeze all supplied tuning, including nested critics and checkers.
        # Parameters of controllers not used by this trial do not affect it.
        if node == 'controller_server':
            runtime_expected = expected
            if TURNAROUND_GOAL_CHECKER in actual.get('goal_checker_plugins', []):
                runtime_expected = with_turnaround_checker(runtime_expected)
            if ALIGNMENT_GOAL_CHECKER in actual.get('goal_checker_plugins', []):
                runtime_expected = with_alignment_checker(runtime_expected)
            wanted = flatten_parameters(runtime_expected[node]['ros__parameters'])
            unused = set(wanted['controller_plugins']) - {controller}
            wanted = {k: v for k, v in wanted.items()
                      if not any(k.startswith(other + '.') for other in unused)}
        for field, value in wanted.items():
            if field not in actual or actual[field] != value:
                raise ValueError(f'Runtime parameter mismatch: {node}.{field}')


def snapshot_parameters(node, name, timeout_s=15.):
    """Query known parameter services directly while servicing recorder callbacks.

    A CLI daemon's node-name cache can lag stack restarts. Service discovery and
    responses instead share one bounded monotonic deadline, without a subprocess.
    """
    import rclpy
    from rcl_interfaces.srv import GetParameters, ListParameters
    from rclpy.parameter import parameter_value_to_python
    deadline = time.monotonic() + timeout_s
    clients = []

    def call(service_type, suffix, request):
        client = node.create_client(service_type, '/' + name + '/' + suffix)
        clients.append(client)
        while not client.service_is_ready():
            if time.monotonic() >= deadline:
                raise RuntimeError('Parameter service unavailable: ' + name + '/' + suffix)
            rclpy.spin_once(node, timeout_sec=.01)
        pending = client.call_async(request)
        while not pending.done():
            if time.monotonic() >= deadline:
                raise RuntimeError('Parameter response timed out: ' + name + '/' + suffix)
            rclpy.spin_once(node, timeout_sec=.01)
        return pending.result()

    try:
        names = sorted(call(ListParameters, 'list_parameters', ListParameters.Request()).result.names)
        request = GetParameters.Request(); request.names = names
        response = call(GetParameters, 'get_parameters', request)
        if len(response.values) != len(names):
            raise RuntimeError('Incomplete parameter response: ' + name)
        return {'/' + name: {'ros__parameters': {
            key: parameter_value_to_python(value) for key, value in zip(names, response.values)}}}
    finally:
        for client in clients:
            node.destroy_client(client)


def summarize(session):
    # Import lazily so config/path helpers remain available to standalone tools.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dwvp_access_metrics import summarize_session
    return summarize_session(session, tracking_errors)


def session_status(session):
    """List trials in frozen order without modifying the session or importing ROS."""
    session = Path(session)
    manifest = json.loads((session / 'manifest.json').read_text())
    rows = []
    for trial in manifest['trials']:
        folder = session / 'runs' / trial['id']
        if (folder / 'result.json').exists():
            state = json.loads((folder / 'result.json').read_text())['status']
        elif folder.exists():
            state = 'incomplete'
        elif manifest['map_origin'] is None:
            state = 'needs_origin'
        elif manifest['paths'][trial['task']].get('file') is None:
            state = 'needs_route'
        else:
            state = 'pending'
        rows.append({'trial': trial['id'], 'status': state, 'direction': trial.get('direction', 'forward')})
    return rows


def pose_path_message(reference, frame, stamp):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    msg = RosPath()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    for x, y, yaw in reference:
        p = PoseStamped(); p.header = msg.header
        p.pose.position.x = float(x); p.pose.position.y = float(y)
        p.pose.orientation.z = math.sin(yaw / 2); p.pose.orientation.w = math.cos(yaw / 2)
        msg.poses.append(p)
    return msg


def preview(session, trial_id):
    """Publish only the frozen reference for RViz. No action client or velocity publisher."""
    import rclpy
    from nav_msgs.msg import Path as RosPath
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    manifest, trial, reference = load_trial(session, trial_id)
    rclpy.init()
    node = Node('dwvp_access_preview')
    publisher = node.create_publisher(RosPath, '/dwvp_access/reference_path',
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    start_publisher = node.create_publisher(PoseStamped, '/dwvp_access/start_pose',
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        start_publisher.publish(pose_path_message([start_pose(manifest, trial)], manifest['frame'], node.get_clock().now().to_msg()).poses[0])
        publisher.publish(pose_path_message(reference, manifest['frame'], node.get_clock().now().to_msg()))
        print('Reference published on /dwvp_access/reference_path. No motion goal sent. Ctrl-C to close.', flush=True)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C can surface as either exception, depending on which signal handler runs first.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def scan_quality(scan, received, stamp):
    """Validate sensor timing and usable ranges; static laser TF has no age test."""
    source = scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9
    receive_age, source_age = stamp - received, stamp - source
    if not scan.header.frame_id or not (0 <= receive_age <= .5 and source_age_is_fresh(source_age, .5)):
        raise RuntimeError(f'A fresh laser scan with source timestamps and a frame is required '
                           f'(receive age {receive_age:.3f} s, source age {source_age:.3f} s; '
                           f'future tolerance {CLOCK_FUTURE_TOLERANCE_S:.3f} s)')
    ranges = np.asarray(scan.ranges)
    usable = (np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)) | np.isposinf(ranges)
    if not len(ranges) or not np.any(usable) or not (0 <= scan.range_min < scan.range_max):
        raise RuntimeError('Laser scan contains no usable range measurements')
    return {'frame': scan.header.frame_id, 'receive_age_s': receive_age, 'source_age_s': source_age}


def scan_minimum_range(scan):
    ranges = np.asarray(scan.ranges)
    usable = ranges[np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)]
    if len(usable):
        return float(usable.min())
    return math.inf if np.any(np.isposinf(ranges)) else math.nan


def wait_for_settled_pose(sample, spin_once, timeout, *, clock=time.monotonic):
    """Observe a continuous stationary interval after the action; send no goal.

    sample() must reject stale input and return (pose, stopped). A success action
    response can precede delivery of its final TF and the smoother's deceleration.
    """
    begin = clock()
    stable_since = None
    reason = 'No fresh stop observation'
    while clock() - begin < timeout:
        spin_once()
        try:
            pose, stopped = sample()
            if not stopped:
                raise RuntimeError('Robot or smoothed command is still moving')
            if stable_since is None:
                stable_since = clock()
            if clock() - stable_since >= .5:
                return pose
        except Exception as exc:
            reason = str(exc)
            stable_since = None
    raise RuntimeError(f'Post-goal stop not verified within {timeout:.1f} s: {reason}')


def run(session, trial_id, base_frame='base_link', odom_topic='/omni_base_controller/wheel_odom', preflight_only=False,
        *, transfer=None, start_capture_file=None):
    import uuid
    import rclpy
    from action_msgs.msg import GoalStatus
    from action_msgs.srv import CancelGoal
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from nav2_msgs.action import FollowPath
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from tf2_ros import Buffer, TransformListener
    from unique_identifier_msgs.msg import UUID
    from sensor_msgs.msg import LaserScan
    from diagnostic_msgs.msg import DiagnosticArray
    import yaml

    session = Path(session)
    manifest, trial, reference = load_trial(session, trial_id)
    frozen_trial = dict(trial)
    capture_packet = None
    endpoint_policy = 'fixed_tolerance'
    if start_capture_file is not None:
        if transfer is not None:
            raise ValueError('A transfer cannot reanchor a scored trial')
        capture_packet = json.loads(Path(start_capture_file).read_text())
        if (capture_packet.get('trial_id') != trial_id
                or capture_packet.get('manifest_sha256') != digest(session/'manifest.json')):
            raise ValueError('Start capture does not match the frozen trial/session')
        trial, reference = reanchor_trial(manifest, trial, reference, capture_packet['capture']['pose'])
    if transfer is not None:
        if preflight_only:
            raise ValueError('Transfer recording cannot be combined with preflight_only')
        endpoint_policy = transfer.get('endpoint_policy', 'fixed_tolerance')
        if endpoint_policy not in ('fixed_tolerance', 'reanchor_after_stop'):
            raise ValueError('Unknown transfer endpoint policy')
        manifest, trial = copy.deepcopy(manifest), dict(trial)
        reference = np.asarray(transfer['path'], dtype=float)
        transfer_goal_checker(transfer, reference)
        if (reference.shape == (2, 3) and np.isfinite(reference).all()
                and np.linalg.norm(reference[1, :2]-reference[0, :2]) <= 1e-6):
            pass  # An unscored, in-place goal-heading adjustment by DWVP.
        else:
            reference = validate_path(reference)
        initial = np.asarray(transfer['start'], dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError('Finite transfer start required')
        manifest['starts'][trial['task']]['map_pose'] = initial.tolist()
        trial['start_pose'] = initial.tolist()
        trial['controller'] = 'DWVP'
    temporary = tempfile.TemporaryDirectory(prefix='dwvp-preflight-') if preflight_only else None
    folder = Path(temporary.name) if temporary else session / 'runs' / trial_id
    if transfer is not None:
        folder = Path(transfer['output'])
        # Never let relocation data overwrite or masquerade as an experiment.
        if session.resolve() / 'runs' in folder.resolve().parents:
            raise ValueError('Transfer output must be outside the experimental runs directory')
    if temporary is None:
        folder.mkdir(parents=True, exist_ok=False)
    np.savetxt(folder / 'reference.csv', reference, delimiter=',', header='x,y,yaw', comments='')
    metadata = {'trial': frozen_trial if capture_packet is not None else trial,
                'manifest_sha256': digest(session/'manifest.json'), 'params_sha256': trial['params_sha256'],
                'clock': 'ROS time; event receive stamps',
                'reference_policy': 'per_trial_current_pose' if capture_packet is not None else 'session_fixed',
                'endpoint_policy': endpoint_policy,
                'reference_sha256': digest(folder/'reference.csv')}
    if capture_packet is not None:
        write_json(folder/'start_capture.json', capture_packet)
        metadata.update(actual_start_pose=start_pose(manifest, trial).tolist(),
                        start_capture_sha256=digest(folder/'start_capture.json'))
        metadata['start_policy'] = capture_packet.get('start_policy', 'align_then_current_pose')
        metadata['source_direction'] = frozen_trial.get('direction', 'forward')
    write_json(folder/'trial.json', metadata)
    # Keep the context alive on Ctrl-C until this client's goal is canceled.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('dwvp_access_recorder')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    client = ActionClient(node, FollowPath, '/follow_path')
    files = []
    latest = {}
    latest_scan = []
    config = yaml.safe_load((session / trial['params_file']).read_text())
    scan_topic = config['local_costmap']['local_costmap']['ros__parameters']['voxel_layer']['scan']['topic']
    command_first = {}
    recording = False
    start = node.get_clock().now().nanoseconds / 1e9

    def now():
        return node.get_clock().now().nanoseconds / 1e9

    def writer(name, header):
        f = (folder / name).open('w', newline='')
        files.append(f)
        w = csv.writer(f); w.writerow(header)
        return w

    event_writers = {k: writer(k + '.csv', ['stamp_s', 'source_stamp_s', 'vx', 'vy', 'omega', 'receive_monotonic_s'])
                     for k in ('raw', 'applied', 'odom')}
    odom_pose_writer = writer('odom_pose.csv', ['stamp_s', 'source_stamp_s', 'frame', 'child_frame',
        'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    scan_writer = writer('scan_ranges.csv', ['stamp_s', 'source_stamp_s', 'frame', 'angle_min',
        'angle_increment', 'range_min', 'range_max', 'ranges_json'])

    def callback(key, msg):
        received_monotonic = time.monotonic()
        twist = msg.twist.twist if key == 'odom' else msg
        source = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9 if key == 'odom' else None
        latest[key] = (now(), twist, source)
        if recording and key in ('raw', 'applied'):
            command_first.setdefault(key, latest[key][0])
        event_writers[key].writerow([latest[key][0], source, twist.linear.x, twist.linear.y, twist.angular.z, received_monotonic])
        if key == 'odom':
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            odom_pose_writer.writerow([latest[key][0], source, msg.header.frame_id, msg.child_frame_id,
                p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    timing_writer = writer('timing.csv', ['receive_stamp_s', 'stamp_s', 'controller', 'sequence', 'compute_time_ms', 'success'])

    def timing_callback(msg):
        for item in msg.status:
            if item.name != trial['controller']:
                continue
            values = {v.key: v.value for v in item.values}
            try:
                duration_ms = float(values.get('duration_ns', 'nan')) / 1e6
            except ValueError:
                duration_ms = math.nan
            timing_writer.writerow([now(), msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                item.name, values.get('sequence', ''), duration_ms, values.get('success', '')])

    def scan_callback(msg):
        latest_scan[:] = [msg, now()]
        # Preserve bearing-specific evidence; a single minimum cannot diagnose a blind spot.
        scan_writer.writerow([latest_scan[1], msg.header.stamp.sec + msg.header.stamp.nanosec/1e9,
            msg.header.frame_id, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
            json.dumps([float(r) if math.isfinite(r) else None for r in msg.ranges], separators=(',', ':'))])

    subscriptions = [node.create_subscription(DiagnosticArray, '/dwvp_access/controller_timing', timing_callback, 1000),
                     node.create_subscription(Twist, '/cmd_vel_nav', lambda m: callback('raw', m), 100),
                     node.create_subscription(Twist, '/omni_base_controller/cmd_vel', lambda m: callback('applied', m), 100),
                     node.create_subscription(Odometry, odom_topic, lambda m: callback('odom', m), qos_profile_sensor_data),
                     node.create_subscription(LaserScan, scan_topic, scan_callback, qos_profile_sensor_data)]
    track = writer('tracking.csv', ['t', 'stamp_s', 'x', 'y', 'yaw', 'tf_age_s', 'raw_age_s', 'applied_age_s', 'odom_age_s', 'odom_source_age_s', 'scan_age_s', 'scan_source_age_s', 'scan_min_range_m', 'speed_m_s'])
    result = {'status': 'setup_failed', 'success': False, 'direction': trial.get('direction', 'forward'),
              'maximum_future_source_skew_s': CLOCK_FUTURE_TOLERANCE_S,
              'reference_policy': metadata['reference_policy'],
              'endpoint_policy': endpoint_policy,
              'goal_checker_id': transfer_goal_checker(transfer, reference) if transfer is not None else 'general_goal_checker'}
    if capture_packet is not None:
        result['start_policy'] = metadata['start_policy']
        result['source_direction'] = metadata['source_direction']
    if transfer is not None:
        result['purpose'] = 'reposition_only_not_an_experiment_trial'
        result['direction'] = 'positioning'
    goal_handle = None
    accepted = response = None
    goal_uuid = UUID(uuid=list(uuid.uuid4().bytes))

    def pose():
        tf = buffer.lookup_transform(manifest['frame'], base_frame, rclpy.time.Time())
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec / 1e9
        return [tf.transform.translation.x, tf.transform.translation.y, yaw, now() - stamp]

    def odom_source_age(stamp):
        return stamp - latest['odom'][2] if 'odom' in latest else math.inf

    def cancel_and_wait():
        """Request this goal's cancellation and separately confirm its terminal state."""
        nonlocal goal_handle, response
        result['cancellation_requested'] = True
        result['cancellation_confirmed'] = False
        if goal_handle is None and accepted is not None and accepted.done():
            goal_handle = accepted.result()
        if goal_handle is not None and goal_handle.accepted:
            cancellation = goal_handle.cancel_goal_async()
        else:
            # A timed-out acceptance response does not imply that the server
            # rejected the goal. Address cancellation by its original UUID.
            cancel_client = node.create_client(CancelGoal, '/follow_path/_action/cancel_goal')
            if not cancel_client.wait_for_service(timeout_sec=2):
                result['cancellation_error'] = 'Cancellation service unavailable'
                return
            request = CancelGoal.Request(); request.goal_info.goal_id = goal_uuid
            cancellation = cancel_client.call_async(request)
        stop_deadline = time.monotonic() + 5
        while time.monotonic() < stop_deadline:
            rclpy.spin_once(node, timeout_sec=.01)
            if goal_handle is None and accepted is not None and accepted.done():
                goal_handle = accepted.result()
                if goal_handle.accepted:
                    # Retry after a late acceptance, covering a cancellation
                    # request that reached the server before goal registration.
                    cancellation = goal_handle.cancel_goal_async()
            if goal_handle is not None and goal_handle.accepted and response is None:
                response = goal_handle.get_result_async()
            if cancellation.done():
                cancelled = cancellation.result()
                result['cancel_response_code'] = int(cancelled.return_code)
                result['cancel_acknowledged'] = bool(cancelled.goals_canceling)
            if response is not None and response.done():
                terminal = response.result().status
                result['cancellation_terminal_status'] = int(terminal)
                result['cancellation_confirmed'] = terminal in (
                    GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED)
                if result['cancellation_confirmed']:
                    return
        result['cancellation_error'] = 'Action terminal state not confirmed within 5 seconds'

    try:
        if not client.wait_for_server(timeout_sec=10):
            raise RuntimeError('FollowPath action server unavailable')
        runtime_files = {}
        # Snapshot queries continue spinning so queued old sensor/command samples
        # cannot silently acquire fresh receive timestamps after a blocking query.
        for name in ('controller_server', 'velocity_smoother'):
            snapshot = snapshot_parameters(node, name)
            runtime_files[name] = folder / (name + '_runtime.yaml')
            runtime_files[name].write_text(yaml.safe_dump(snapshot))
        verify_runtime_parameters(session / trial['params_file'], runtime_files, trial['controller'])
        deadline = time.monotonic() + 10
        current = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            try:
                current = pose()
            except Exception:
                continue
            if 'odom' in latest and 0 <= now() - latest['odom'][0] <= .2 and source_age_is_fresh(odom_source_age(now())) and source_age_is_fresh(current[3]):
                break
        if current is None or 'odom' not in latest or not (source_age_is_fresh(current[3]) and 0 <= now() - latest['odom'][0] <= .2 and source_age_is_fresh(odom_source_age(now()))):
            raise RuntimeError('Fresh map pose and odometry are required')
        check_start_pose(current, manifest, trial)
        scan_state = None
        deadline = time.monotonic() + 5
        scan_error = 'No laser scan received on ' + scan_topic
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if not latest_scan:
                continue
            try:
                scan, received = latest_scan
                scan_state = scan_quality(scan, received, now())
                buffer.lookup_transform(base_frame, scan.header.frame_id, rclpy.time.Time.from_msg(scan.header.stamp))
                break
            except Exception as exc:
                scan_state = None
                scan_error = str(exc)
        if scan_state is None:
            raise RuntimeError('Laser preflight failed: ' + scan_error)
        # Sensor discovery can take several seconds; recheck the start pose afterward.
        current = pose()
        if not (source_age_is_fresh(current[3]) and 0 <= now() - latest['odom'][0] <= .2 and source_age_is_fresh(odom_source_age(now()))):
            raise RuntimeError('Map pose or odometry became stale during sensor preflight')
        check_start_pose(current, manifest, trial)
        result['preflight'] = {'scan_topic': scan_topic, 'scan': scan_state, 'base_frame': base_frame,
                               'map_pose': current[:3], 'expected_start_pose': start_pose(manifest, trial).tolist(), 'odom_topic': odom_topic}
        if preflight_only:
            result.pop('success')
            result.update(status='ready', ready=True, motion_goal_sent=False)
            return result
        msg = pose_path_message(reference, manifest['frame'], node.get_clock().now().to_msg())
        goal = FollowPath.Goal(); goal.path = msg
        goal.controller_id = trial['controller']; goal.goal_checker_id = result['goal_checker_id']
        accepted = client.send_goal_async(goal, goal_uuid=goal_uuid)
        rclpy.spin_until_future_complete(node, accepted, timeout_sec=10)
        if not accepted.done():
            raise RuntimeError('Goal acceptance timed out')
        goal_handle = accepted.result()
        if not goal_handle.accepted:
            raise RuntimeError('Goal rejected')
        start = now()
        result['start_stamp_s'] = start
        recording = True
        deadline = time.monotonic() + manifest['timeout_s']
        response = goal_handle.get_result_async()
        tick = time.monotonic()
        while not response.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.005)
            if time.monotonic() >= tick:
                stamp = now()
                try:
                    p = pose()
                except Exception:
                    p = [math.nan, math.nan, math.nan, math.inf]
                ages = [stamp - latest[k][0] if k in latest and (k == 'odom' or k in command_first)
                        else math.inf for k in ('raw', 'applied', 'odom')]
                scan_values = [math.inf, math.inf, math.nan]
                if latest_scan:
                    scan, received = latest_scan
                    scan_values = [stamp-received, stamp-(scan.header.stamp.sec+scan.header.stamp.nanosec/1e9),
                                   scan_minimum_range(scan)]
                speed = math.hypot(latest['odom'][1].linear.x, latest['odom'][1].linear.y) if 'odom' in latest else math.nan
                track.writerow([stamp - start, stamp, *p, *ages, odom_source_age(stamp), *scan_values, speed])
                tick += 1 / manifest['control_frequency_hz']
        duration = now() - start
        result.update(duration_s=duration,
                      missing_command_prefix_s=max(command_first.values()) - start if len(command_first) == 2 else duration)
        if not response.done():
            result.update(status='timeout', duration_s=duration)
            cancel_and_wait()
        else:
            status = response.result().status
            result['action_status'] = int(status)
            settled = False
            if status == GoalStatus.STATUS_SUCCEEDED:
                smooth = config['velocity_smoother']['ros__parameters']
                stopping_time = max(abs(v/a) for v, a in zip(smooth['max_velocity'], smooth['max_decel']))
                settle_timeout = max(5., stopping_time + smooth.get('velocity_timeout', .5) + 2.)
                settle_begin = time.monotonic()
                settle_writer = writer('settling.csv', ['stamp_s', 'x', 'y', 'yaw', 'tf_age_s',
                    'odom_age_s', 'odom_source_age_s', 'speed_m_s', 'yaw_rate_rad_s', 'applied_zero', 'fresh'])

                def stop_sample():
                    p = pose()
                    stamp = now()
                    odom = latest.get('odom')
                    velocity = odom[1] if odom else None
                    speed = math.hypot(velocity.linear.x, velocity.linear.y) if velocity else math.inf
                    yaw_rate = abs(velocity.angular.z) if velocity else math.inf
                    odom_age = stamp - odom[0] if odom else math.inf
                    source_age = odom_source_age(stamp)
                    applied = latest.get('applied')
                    # The smoother may stop publishing after reaching zero;
                    # fresh wheel odometry must still confirm a physical stop.
                    zero = bool(applied and 'applied' in command_first and all(
                        math.isfinite(v) and abs(v) < 1e-6 for v in
                        (applied[1].linear.x, applied[1].linear.y, applied[1].angular.z)))
                    fresh = bool(np.isfinite(p[:3]).all() and source_age_is_fresh(p[3])
                                 and 0 <= odom_age <= .2 and source_age_is_fresh(source_age)
                                 and math.isfinite(speed) and math.isfinite(yaw_rate))
                    settle_writer.writerow([stamp, *p, odom_age, source_age, speed, yaw_rate, zero, fresh])
                    if not fresh:
                        raise RuntimeError('Fresh TF and wheel odometry required after goal completion')
                    return p, zero and speed <= .005 and yaw_rate <= .01

                try:
                    wait_for_settled_pose(stop_sample, lambda: rclpy.spin_once(node, timeout_sec=.02), settle_timeout)
                    settled = True
                except RuntimeError as exc:
                    result['error'] = str(exc)
                result['settling'] = dict(verified=settled, duration_s=time.monotonic()-settle_begin,
                                          timeout_s=settle_timeout, stationary_interval_s=.5)
            # Scored runs and fixed-start transfers keep their original endpoint
            # limits. A transfer followed by reanchoring needs a successful action
            # and a fresh stopped pose; its residual is recorded, not rejected.
            last = pose()
            fresh = bool(np.isfinite(last[:3]).all() and source_age_is_fresh(last[3]))
            within = fresh and np.linalg.norm(np.asarray(last[:2]) - reference[-1, :2]) <= manifest['xy_tolerance_m'] and abs(float(wrap(last[2] - reference[-1, 2]))) <= manifest['yaw_tolerance_rad']
            success = (status == GoalStatus.STATUS_SUCCEEDED and settled and fresh
                       and (within or endpoint_policy == 'reanchor_after_stop'))
            result.update(status='succeeded' if success else 'failed',
                          action_status=status, success=bool(success),
                          final_pose=last[:3], final_pose_within_tolerances=bool(within),
                          final_pose_fresh=fresh,
                          duration_s=duration, final_position_error_m=float(np.linalg.norm(np.asarray(last[:2]) - reference[-1, :2])),
                          final_yaw_error_rad=abs(float(wrap(last[2] - reference[-1, 2]))))
            if status == GoalStatus.STATUS_SUCCEEDED and settled and fresh and not within and not success:
                result['failure_reason'] = 'endpoint_tolerance_exceeded'
            if not success and 'error' not in result:
                result['error'] = (f'Final stopped pose outside tolerances or action unsuccessful: '
                    f'action_status={status}, position={result["final_position_error_m"]:.4f} m '
                    f'(limit {manifest["xy_tolerance_m"]:.4f}), '
                    f'yaw={result["final_yaw_error_rad"]:.4f} rad (limit {manifest["yaw_tolerance_rad"]:.4f})')
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error', error=str(exc))
        if recording:
            result.update(duration_s=now() - start,
                          missing_command_prefix_s=max(command_first.values()) - start if len(command_first) == 2 else now() - start)
        if accepted is not None and (goal_handle is None or goal_handle.accepted):
            try:
                cancel_and_wait()
            except Exception as cancel_error:
                result.update(cancellation_confirmed=False, cancellation_error=str(cancel_error))
        raise
    finally:
        drain_deadline = time.monotonic() + .15
        try:
            while time.monotonic() < drain_deadline:
                rclpy.spin_once(node, timeout_sec=.01)
        except (Exception, KeyboardInterrupt) as exc:
            result['timing_drain_error'] = str(exc)
        write_json(folder / 'result.json', result)
        for f in files: f.close()
        node.destroy_node()
        rclpy.try_shutdown()
        if temporary:
            temporary.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--params', type=Path, required=True)
    placement = p.add_mutually_exclusive_group()
    placement.add_argument('--origin', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    placement.add_argument('--start-from-current', action='store_true',
                           help='Freeze the current localized pose as the start of each E1 condition')
    p.add_argument('--base-frame', default='base_link')
    p.add_argument('--odom-topic', default='/omni_base_controller/wheel_odom')
    p.add_argument('--environment-path', type=Path, help='NavFn CSV in map coordinates')
    p.add_argument('--config', type=Path, help='Experiment geometry, common tuning and metric settings')
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--bidirectional', action='store_true', help='Score alternating forward/reverse legs; freeze their starts')
    p.add_argument('--conditions', nargs='+', choices=['E1', *CONDITIONS])
    p.add_argument('--repeats', type=int, nargs='+', help='Freeze only these repetitions (1..5)')
    for command in ('run', 'preflight'):
        p = commands.add_parser(command)
        p.add_argument('--session', type=Path, required=True); p.add_argument('--trial', required=True)
        p.add_argument('--base-frame', default='base_link'); p.add_argument('--odom-topic', default='/omni_base_controller/wheel_odom')
        p.add_argument('--start-capture', type=Path, help='Batch-captured E1 start; preserve local experimental geometry')
    p = commands.add_parser('parameters')
    p.add_argument('--session', type=Path, required=True); p.add_argument('--trial', required=True)
    p = commands.add_parser('preview')
    p.add_argument('--session', type=Path, required=True); p.add_argument('--trial', required=True)
    p = commands.add_parser('status'); p.add_argument('--session', type=Path, required=True)
    p = commands.add_parser('summarize'); p.add_argument('--session', type=Path, required=True)
    p = commands.add_parser('stationary-noise', help='Passively record map pose while manually stopped')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration', type=float, default=30.)
    p.add_argument('--frequency', type=float, default=30.)
    p.add_argument('--frame', default='map'); p.add_argument('--base-frame', default='base_link')
    p.add_argument('--odom-topic', default='/omni_base_controller/wheel_odom')
    p.add_argument('--max-age', type=float, default=.2)
    args = parser.parse_args()
    if args.command == 'prepare':
        captured = None
        if args.start_from_current:
            if args.output.exists():
                raise FileExistsError(f'Session already exists: {args.output}; choose a new name')
            from dwvp_access_runtime import capture_start
            captured = capture_start(args.base_frame, args.odom_topic)
        result = prepare(args.output, args.params, args.origin, args.environment_path, args.seed, args.config,
                         current_start=captured['pose'] if captured else None,
                         bidirectional=args.bidirectional, conditions=args.conditions, repeats=args.repeats)
        if captured:
            write_json(args.output / 'start_capture.json', captured)
            print('Frozen robot start [map x, y, yaw]: ' + str(captured['pose']))
        print(f"Prepared {len(result['trials'])} unrecorded trials in {args.output}")
    elif args.command == 'stationary-noise':
        from dwvp_access_noise import record_noise
        result = record_noise(args.output, args.duration, args.frequency, args.frame,
                              args.base_frame, args.odom_topic, args.max_age)
        print(json.dumps(result, indent=2))
        if not result['stationary_verified']:
            raise SystemExit('Stationarity was not verified; inspect noise_samples.csv')
    elif args.command in ('run', 'preflight'):
        print(json.dumps(run(args.session, args.trial, args.base_frame, args.odom_topic,
                             preflight_only=args.command == 'preflight', start_capture_file=args.start_capture), indent=2))
    elif args.command == 'parameters':
        _, trial, _ = load_trial(args.session, args.trial)
        print((args.session / trial['params_file']).resolve())
    elif args.command == 'preview':
        preview(args.session, args.trial)
    elif args.command == 'status':
        for row in session_status(args.session):
            print(f"{row['trial']:<40} {row['direction']:<7} {row['status']}")
    else:
        result = summarize(args.session)
        print(f"Recorded {result['recorded']}/{result['planned']}; pending {result['pending']}")


if __name__ == '__main__':
    main()
