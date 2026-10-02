#!/usr/bin/env python3
"""Prepare 80 fixed-path trials, record one ROS trial, and summarize completed data.

Preparation and analysis do not import ROS. Only the explicit ``run`` subcommand
sends a FollowPath goal. A session fixes one map origin for all methods.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def arclength(xy):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def canonical_path(name, independent=False):
    if name == 'path1':
        xy = np.vstack((np.c_[np.linspace(0, 1, 101), np.zeros(101)],
                        np.c_[np.ones(100), np.linspace(.01, 1, 100)]))
        delta = np.diff(xy, axis=0)
        yaw = np.r_[np.arctan2(delta[:, 1], delta[:, 0]), np.pi / 2]
    elif name == 'path2':
        x = np.linspace(0, 1.5, 501)
        xy = np.c_[x, .75 * (1 - np.cos(2 * np.pi * 1.5 * x / 1.5))]
        s = arclength(xy)
        samples = np.linspace(0, s[-1], 501)
        xy = np.c_[np.interp(samples, s, xy[:, 0]), np.interp(samples, s, xy[:, 1])]
        yaw = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    else:
        raise ValueError(name)
    if independent:
        u = arclength(xy) / arclength(xy)[-1]
        yaw = np.zeros(len(xy)) if name == 'path1' else (np.pi / 2) * (3 * u**2 - 2 * u**3)
    return np.c_[xy, yaw]


def validate_path(path):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('A path must have at least two finite x,y,yaw rows')
    if np.any(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) <= 1e-9):
        raise ValueError('Duplicate position samples are not supported')
    return path


def prepare(output, params, origin=None, obstacle_path=None, seed=20261003):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'paths').mkdir()
    shutil.copy2(params, output / 'nav2_params.yaml')
    paths = {}
    for independent in (False, True):
        for name in ('path1', 'path2'):
            key = ('B_' if independent else 'A_') + name
            file = output / 'paths' / (key + '.csv')
            np.savetxt(file, canonical_path(name, independent), delimiter=',', header='x,y,yaw', comments='')
            paths[key] = {'file': str(file.relative_to(output)), 'sha256': digest(file)}
    if obstacle_path is not None:
        xytheta = validate_path(np.loadtxt(obstacle_path, delimiter=',', skiprows=1))
        file = output / 'paths/A_obstacle.csv'
        np.savetxt(file, xytheta, delimiter=',', header='x,y,yaw', comments='')
        paths['A_obstacle'] = {'file': str(file.relative_to(output)), 'sha256': digest(file)}
    else:
        paths['A_obstacle'] = {'file': None, 'status': 'needs_surveyed_static_obstacle_route'}
    trials = []
    rng = random.Random(seed)
    for repeat in range(1, 6):
        block = []
        for task in ('A_path1', 'A_path2', 'A_obstacle', 'B_path1', 'B_path2'):
            controllers = ('RPP', 'DWPP', 'MPPI', 'DWVP') if task.startswith('A_') else ('MPPI', 'DWVP')
            for controller in controllers:
                block.append({'id': f'{task}_{controller}_r{repeat}', 'task': task,
                              'controller': controller, 'repeat': repeat})
        rng.shuffle(block)
        trials.extend(block)
    manifest = {'schema_version': 1, 'seed': seed, 'frame': 'map', 'map_origin': origin,
                'params_file': 'nav2_params.yaml', 'params_sha256': digest(output / 'nav2_params.yaml'),
                'control_frequency_hz': 30, 'xy_tolerance_m': .1, 'yaw_tolerance_rad': .3,
                'timeout_s': 120, 'paths': paths, 'trials': trials,
                'status': 'prepared_unrecorded'}
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_trial(session, trial_id):
    session = Path(session)
    manifest = json.loads((session / 'manifest.json').read_text())
    trial = next((t for t in manifest['trials'] if t['id'] == trial_id), None)
    if trial is None:
        raise ValueError('Unknown trial ID')
    if manifest['map_origin'] is None:
        raise ValueError('Prepare a session with a surveyed --origin X Y YAW before recording')
    record = manifest['paths'][trial['task']]
    if record.get('file') is None:
        raise ValueError('The static-obstacle route has not been surveyed and supplied')
    if digest(session / record['file']) != record['sha256'] or digest(session / manifest['params_file']) != manifest['params_sha256']:
        raise ValueError('Frozen path/configuration changed; prepare a new session')
    path = validate_path(np.loadtxt(session / record['file'], delimiter=',', skiprows=1))
    ox, oy, angle = manifest['map_origin']
    c, s = math.cos(angle), math.sin(angle)
    path[:, :2] = path[:, :2] @ np.array([[c, s], [-s, c]]) + [ox, oy]
    path[:, 2] = wrap(path[:, 2] + angle)
    return manifest, trial, path


def tracking_errors(poses, path):
    """Project position onto segments and use that same location for yaw error."""
    a, delta = path[:-1, :2], np.diff(path[:, :2], axis=0)
    lensq = np.sum(delta * delta, axis=1)
    errors = []
    for pose in poses:
        t = np.clip(np.sum((pose[:2] - a) * delta, axis=1) / lensq, 0, 1)
        closest = a + t[:, None] * delta
        dist = np.linalg.norm(closest - pose[:2], axis=1)
        k = int(np.argmin(dist))
        yaw = path[k, 2] + t[k] * wrap(path[k + 1, 2] - path[k, 2])
        errors.append((dist[k], abs(float(wrap(pose[2] - yaw)))))
    return np.asarray(errors)


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
        wanted = flatten_parameters(expected[node]['ros__parameters'])
        # Freeze all supplied tuning, including nested critics and checkers.
        # Parameters of controllers not used by this trial do not affect it.
        if node == 'controller_server':
            unused = set(wanted['controller_plugins']) - {controller}
            wanted = {k: v for k, v in wanted.items()
                      if not any(k.startswith(other + '.') for other in unused)}
        snapshot = yaml.safe_load(Path(runtime_files[node]).read_text())
        actual = snapshot.get('/' + node, snapshot.get(node))
        if actual is None:
            raise ValueError('Runtime snapshot is missing node ' + node)
        actual = flatten_parameters(actual['ros__parameters'])
        for field, value in wanted.items():
            if field not in actual or actual[field] != value:
                raise ValueError(f'Runtime parameter mismatch: {node}.{field}')


def command_metrics(file, start, duration, frequency):
    """Command-space diagnostics; receipt jitter is not physical acceleration."""
    if not Path(file).exists() or start is None or duration is None:
        return {}
    events = np.atleast_1d(np.genfromtxt(file, delimiter=',', names=True))
    events = events[(events['stamp_s'] >= start) & (events['stamp_s'] <= start + duration)]
    if not len(events):
        return {'samples': 0}
    values = np.c_[events['vx'], events['vy'], events['omega']]
    velocity_excess = np.maximum(np.abs(values) / [.22, .22, .6] - 1., 0.)
    out = {'samples': len(events), 'velocity_excess_ratio_max': float(velocity_excess.max()),
           'velocity_excess_samples': int(np.any(velocity_excess > 1e-6, axis=1).sum())}
    if len(values) >= 2:
        delta_ratio = np.abs(np.diff(values, axis=0)) * frequency / [.22, .22, .6]
        out.update(command_increment_excess_ratio_max=float(np.maximum(delta_ratio - 1., 0.).max()),
                   command_increment_excess_samples=int(np.any(delta_ratio > 1. + 1e-6, axis=1).sum()),
                   receive_interval_max_s=float(np.diff(events['stamp_s']).max()),
                   receive_interval_median_s=float(np.median(np.diff(events['stamp_s']))))
    if len(values) >= 3:
        jerk = np.diff(values, n=2, axis=0) * frequency**2
        out.update(linear_command_jerk_rms_m_s3=float(np.sqrt(np.mean(np.sum(jerk[:, :2]**2, axis=1)))),
                   angular_command_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk[:, 2]**2))))
    return out


def summarize(session):
    session = Path(session)
    manifest = json.loads((session / 'manifest.json').read_text())
    rows = []
    command_diagnostics = {}
    for trial in manifest['trials']:
        folder = session / 'runs' / trial['id']
        if not (folder / 'result.json').exists():
            continue
        result = json.loads((folder / 'result.json').read_text())
        data = np.genfromtxt(folder / 'tracking.csv', delimiter=',', names=True)
        data = np.atleast_1d(data)
        path = np.loadtxt(folder / 'reference.csv', delimiter=',', skiprows=1)
        valid = np.isfinite(data['x']) & np.isfinite(data['y']) & np.isfinite(data['yaw'])
        # Missing/stale measurements are reported, never silently replaced by zeros.
        source_age = data['odom_source_age_s'] if 'odom_source_age_s' in data.dtype.names else np.full(len(data), np.inf)
        measurements_fresh = valid & (data['odom_age_s'] >= 0) & (data['odom_age_s'] <= .2) & (source_age >= 0) & (source_age <= .2) & (data['tf_age_s'] >= 0) & (data['tf_age_s'] <= .2)
        commands_fresh = (data['raw_age_s'] >= 0) & (data['raw_age_s'] <= .2) & (data['applied_age_s'] >= 0) & (data['applied_age_s'] <= .2)
        quality = measurements_fresh & commands_fresh
        grace = measurements_fresh & (data['t'] <= 1 / manifest['control_frequency_hz'])
        missing_prefix = result.get('missing_command_prefix_s')
        if missing_prefix is None:
            ready = np.flatnonzero(commands_fresh)
            missing_prefix = float(data['t'][ready[0]]) if len(ready) else result.get('duration_s')
        row = dict(trial_id=trial['id'], task=trial['task'], controller=trial['controller'],
                   repeat=trial['repeat'], status=result['status'], success=result.get('success', False),
                   duration_s=result.get('duration_s'), samples=len(data), valid_pose_samples=int(valid.sum()),
                   stale_or_missing_samples=int((~quality).sum()), position_rmse_m=None,
                   invalid_after_warmup_samples=int((~(quality | grace)).sum()),
                   missing_command_prefix_s=missing_prefix,
                   position_max_m=None, yaw_rmse_deg=None, yaw_max_deg=None)
        if valid.any():
            errors = tracking_errors(np.c_[data['x'][valid], data['y'][valid], data['yaw'][valid]], path)
            row.update(position_rmse_m=float(np.sqrt(np.mean(errors[:, 0]**2))),
                       position_max_m=float(errors[:, 0].max()),
                       yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors[:, 1]**2)))),
                       yaw_max_deg=float(np.rad2deg(errors[:, 1].max())))
        rows.append(row)
        command_diagnostics[trial['id']] = {stream: command_metrics(folder / (stream + '.csv'),
            result.get('start_stamp_s'), result.get('duration_s'), manifest['control_frequency_hz'])
            for stream in ('raw', 'applied')}
    if rows:
        with (session / 'trial_metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    aggregates = []
    for task, controller in sorted({(r['task'], r['controller']) for r in rows}):
        selected = [r for r in rows if r['task'] == task and r['controller'] == controller]
        complete = [r for r in selected if r['success'] and r['valid_pose_samples'] > 0
                    and r['invalid_after_warmup_samples'] == 0
                    and r['missing_command_prefix_s'] is not None
                    and r['missing_command_prefix_s'] <= 1 / manifest['control_frequency_hz']]
        item = dict(task=task, controller=controller, recorded=len(selected), succeeded=sum(r['success'] for r in selected),
                    valid_successes=len(complete), planned=5)
        for metric in ('duration_s', 'position_rmse_m', 'position_max_m', 'yaw_rmse_deg', 'yaw_max_deg'):
            values = [r[metric] for r in complete if r[metric] is not None]
            item[metric + '_mean'] = float(np.mean(values)) if values else None
            item[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        aggregates.append(item)
    report = {'planned': len(manifest['trials']), 'recorded': len(rows), 'pending': len(manifest['trials']) - len(rows),
              'trials': rows, 'groups': aggregates, 'command_diagnostics': command_diagnostics,
              'note': 'Tracking statistics use valid map poses. Every tracking tick is retained. Group means exclude failed or stale-data runs, allowing at most one initial control period for command startup; missing-prefix duration and all stale samples remain reported. Odometry freshness checks both receive and source time. No controller execution-time or clearance measurements are inferred from these CSVs.'}
    write_json(session / 'summary.json', report)
    return report


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
        rows.append({'trial': trial['id'], 'status': state})
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
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    manifest, _, reference = load_trial(session, trial_id)
    rclpy.init()
    node = Node('dwvp_access_preview')
    publisher = node.create_publisher(RosPath, '/dwvp_access/reference_path',
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        publisher.publish(pose_path_message(reference, manifest['frame'], node.get_clock().now().to_msg()))
        print('Reference published on /dwvp_access/reference_path. No motion goal sent. Ctrl-C to close.', flush=True)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def scan_quality(scan, received, stamp):
    """Validate sensor timing and usable ranges; static laser TF has no age test."""
    source = scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9
    receive_age, source_age = stamp - received, stamp - source
    if not scan.header.frame_id or not (0 <= receive_age <= .5 and 0 <= source_age <= .5):
        raise RuntimeError('A fresh laser scan with source timestamps and a frame is required')
    ranges = np.asarray(scan.ranges)
    usable = (np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)) | np.isposinf(ranges)
    if not len(ranges) or not np.any(usable) or not (0 <= scan.range_min < scan.range_max):
        raise RuntimeError('Laser scan contains no usable range measurements')
    return {'frame': scan.header.frame_id, 'receive_age_s': receive_age, 'source_age_s': source_age}


def run(session, trial_id, base_frame='base_link', odom_topic='/omni_base_controller/wheel_odom', preflight_only=False):
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
    import yaml

    session = Path(session)
    manifest, trial, reference = load_trial(session, trial_id)
    temporary = tempfile.TemporaryDirectory(prefix='dwvp-preflight-') if preflight_only else None
    folder = Path(temporary.name) if temporary else session / 'runs' / trial_id
    if temporary is None:
        folder.mkdir(parents=True, exist_ok=False)
    np.savetxt(folder / 'reference.csv', reference, delimiter=',', header='x,y,yaw', comments='')
    write_json(folder / 'trial.json', {'trial': trial, 'manifest_sha256': digest(session / 'manifest.json'),
                                    'params_sha256': manifest['params_sha256'], 'clock': 'ROS time; event receive stamps'})
    # Keep the context alive on Ctrl-C until this client's goal is canceled.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('dwvp_access_recorder')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    client = ActionClient(node, FollowPath, '/follow_path')
    files = []
    latest = {}
    latest_scan = []
    config = yaml.safe_load((session / manifest['params_file']).read_text())
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

    event_writers = {k: writer(k + '.csv', ['stamp_s', 'source_stamp_s', 'vx', 'vy', 'omega'])
                     for k in ('raw', 'applied', 'odom')}

    def callback(key, msg):
        twist = msg.twist.twist if key == 'odom' else msg
        source = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9 if key == 'odom' else None
        latest[key] = (now(), twist, source)
        if recording and key in ('raw', 'applied'):
            command_first.setdefault(key, latest[key][0])
        event_writers[key].writerow([latest[key][0], source, twist.linear.x, twist.linear.y, twist.angular.z])

    def scan_callback(msg):
        latest_scan[:] = [msg, now()]

    subscriptions = [node.create_subscription(Twist, '/cmd_vel_nav', lambda m: callback('raw', m), 100),
                     node.create_subscription(Twist, '/omni_base_controller/cmd_vel', lambda m: callback('applied', m), 100),
                     node.create_subscription(Odometry, odom_topic, lambda m: callback('odom', m), qos_profile_sensor_data),
                     node.create_subscription(LaserScan, scan_topic, scan_callback, qos_profile_sensor_data)]
    track = writer('tracking.csv', ['t', 'stamp_s', 'x', 'y', 'yaw', 'tf_age_s', 'raw_age_s', 'applied_age_s', 'odom_age_s', 'odom_source_age_s'])
    result = {'status': 'setup_failed', 'success': False}
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
        # Keep servicing subscriptions while the CLI queries node parameters;
        # otherwise queued, old commands acquire fresh receive timestamps later.
        with ThreadPoolExecutor(max_workers=1) as workers:
            for name in ('controller_server', 'velocity_smoother'):
                pending = workers.submit(subprocess.run, ['ros2', 'param', 'dump', '/' + name],
                                         capture_output=True, text=True, timeout=15, check=True)
                while not pending.done():
                    rclpy.spin_once(node, timeout_sec=.01)
                snapshot = pending.result()
                runtime_files[name] = folder / (name + '_runtime.yaml')
                runtime_files[name].write_text(snapshot.stdout)
        verify_runtime_parameters(session / manifest['params_file'], runtime_files, trial['controller'])
        deadline = time.monotonic() + 10
        current = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            try:
                current = pose()
            except Exception:
                continue
            if 'odom' in latest and 0 <= now() - latest['odom'][0] <= .2 and 0 <= odom_source_age(now()) <= .2 and 0 <= current[3] <= .2:
                break
        if current is None or 'odom' not in latest or not (0 <= current[3] <= .2 and 0 <= now() - latest['odom'][0] <= .2 and 0 <= odom_source_age(now()) <= .2):
            raise RuntimeError('Fresh map pose and odometry are required')
        if np.linalg.norm(np.asarray(current[:2]) - reference[0, :2]) > .1 or abs(float(wrap(current[2] - reference[0, 2]))) > .3:
            raise RuntimeError('Reset robot to the frozen reference start pose before this trial')
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
        if not (0 <= current[3] <= .2 and 0 <= now() - latest['odom'][0] <= .2 and 0 <= odom_source_age(now()) <= .2):
            raise RuntimeError('Map pose or odometry became stale during sensor preflight')
        if np.linalg.norm(np.asarray(current[:2]) - reference[0, :2]) > .1 or abs(float(wrap(current[2] - reference[0, 2]))) > .3:
            raise RuntimeError('Robot moved away from the frozen start during sensor preflight')
        result['preflight'] = {'scan_topic': scan_topic, 'scan': scan_state, 'base_frame': base_frame,
                               'map_pose': current[:3], 'odom_topic': odom_topic}
        if preflight_only:
            result.pop('success')
            result.update(status='ready', ready=True, motion_goal_sent=False)
            return result
        msg = pose_path_message(reference, manifest['frame'], node.get_clock().now().to_msg())
        goal = FollowPath.Goal(); goal.path = msg
        goal.controller_id = trial['controller']; goal.goal_checker_id = 'general_goal_checker'
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
                track.writerow([stamp - start, stamp, *p, *ages, odom_source_age(stamp)])
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
            last = pose()
            within = 0 <= last[3] <= .2 and np.linalg.norm(np.asarray(last[:2]) - reference[-1, :2]) <= manifest['xy_tolerance_m'] and abs(float(wrap(last[2] - reference[-1, 2]))) <= manifest['yaw_tolerance_rad']
            result.update(status='succeeded' if status == GoalStatus.STATUS_SUCCEEDED and within else 'failed',
                          action_status=status, success=bool(status == GoalStatus.STATUS_SUCCEEDED and within),
                          duration_s=duration, final_position_error_m=float(np.linalg.norm(np.asarray(last[:2]) - reference[-1, :2])),
                          final_yaw_error_rad=abs(float(wrap(last[2] - reference[-1, 2]))))
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
    p.add_argument('--origin', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    p.add_argument('--obstacle-path', type=Path)
    p.add_argument('--seed', type=int, default=20261003)
    for command in ('run', 'preflight'):
        p = commands.add_parser(command)
        p.add_argument('--session', type=Path, required=True); p.add_argument('--trial', required=True)
        p.add_argument('--base-frame', default='base_link'); p.add_argument('--odom-topic', default='/omni_base_controller/wheel_odom')
    p = commands.add_parser('preview')
    p.add_argument('--session', type=Path, required=True); p.add_argument('--trial', required=True)
    p = commands.add_parser('status'); p.add_argument('--session', type=Path, required=True)
    p = commands.add_parser('summarize'); p.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output, args.params, args.origin, args.obstacle_path, args.seed)
        print(f"Prepared {len(result['trials'])} unrecorded trials in {args.output}")
    elif args.command in ('run', 'preflight'):
        print(json.dumps(run(args.session, args.trial, args.base_frame, args.odom_topic,
                             preflight_only=args.command == 'preflight'), indent=2))
    elif args.command == 'preview':
        preview(args.session, args.trial)
    elif args.command == 'status':
        for row in session_status(args.session):
            print(f"{row['trial']:<28} {row['status']}")
    else:
        result = summarize(args.session)
        print(f"Recorded {result['recorded']}/{result['planned']}; pending {result['pending']}")


if __name__ == '__main__':
    main()
