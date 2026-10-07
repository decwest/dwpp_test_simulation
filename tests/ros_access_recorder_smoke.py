"""Synthetic ROS I/O test. Run only in an isolated container with --network none.

This is a recorder/action test, not evidence about robot/controller performance.
"""
import argparse
from contextlib import nullcontext
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import LaserScan
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import TransformBroadcaster

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--clock-only', action='store_true', help='Check timing, noise observation and one offset-clock trial')
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    config = experiment.render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', experiment.default_config())
    rclpy.init()
    nodes = []
    for name in ('controller_server', 'velocity_smoother'):
        overrides = [Parameter(k, value=v) for k, v in experiment.flatten_parameters(config[name]['ros__parameters']).items()]
        nodes.append(Node(name, parameter_overrides=overrides, automatically_declare_parameters_from_overrides=True))
    robot = Node('synthetic_robot')
    nodes.append(robot)
    broadcaster = TransformBroadcaster(robot)
    odom_pub = robot.create_publisher(Odometry, '/omni_base_controller/wheel_odom', 10)
    scan_pub = robot.create_publisher(LaserScan, '/scan', 10)
    previews = []
    preview_sub = robot.create_subscription(RosPath, '/dwvp_access/reference_path', previews.append,
                                           QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    raw_pub = robot.create_publisher(Twist, '/cmd_vel_nav', 10)
    applied_pub = robot.create_publisher(Twist, '/omni_base_controller/cmd_vel', 10)
    timing_pub = robot.create_publisher(DiagnosticArray, '/dwvp_access/controller_timing', 1000)
    state = {'controller':'DWVP', 'sequence':0, 'path': None, 'index': 0, 'seen_yaw': False, 'mode': 'normal', 'started': None}

    def publish():
        if state['path'] is None:
            x, y, yaw = (0.,0.,0.) if state['mode']=='wrong_start' else (0.,.5,0.)
        else:
            p = state['path'][state['index']].pose
            x, y = p.position.x, p.position.y
            yaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
        settling = state.get('completion_pending', False) and state['mode'] in ('delayed_final_pose', 'settled_outside')
        turning = settling and (state.get('finished') is None or time.monotonic()-state['finished'] < .3)
        if settling:
            yaw = .320 if turning else (.4 if state['mode']=='settled_outside' else .02)
        if state.get('completion_pending'):
            x += state.get('position_residual', 0.)
        skew = {'small_skew': .035, 'future_clock': .1, 'old_clock': -.5}.get(state['mode'], 0.)
        stamp = (robot.get_clock().now() + Duration(seconds=skew)).to_msg()
        tf = TransformStamped(); tf.header.stamp = stamp
        tf.header.frame_id = 'map'; tf.child_frame_id = 'base_link'
        tf.transform.translation.x = x; tf.transform.translation.y = y
        tf.transform.rotation.z = math.sin(yaw / 2); tf.transform.rotation.w = math.cos(yaw / 2)
        broadcaster.sendTransform(tf)
        odom = Odometry(); odom.header.stamp = stamp; odom.header.frame_id = 'odom'
        if state['mode'] == 'stale_source' and state['started'] is not None:
            odom.header.stamp.sec -= 10
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y
        odom.pose.pose.orientation = tf.transform.rotation
        if state['mode'] == 'noise_moving':
            odom.twist.twist.linear.x = .1
        if turning:
            odom.twist.twist.angular.z = .6
        odom_pub.publish(odom)
        if state['mode'] != 'missing_scan':
            scan = LaserScan(); scan.header.frame_id = 'unknown_laser' if state['mode'] == 'bad_scan_frame' else 'base_link'
            scan.header.stamp = stamp
            if state['mode'] == 'stale_scan':
                scan.header.stamp.sec -= 10
            scan.range_min = .05; scan.range_max = 5.; scan.angle_increment = .1
            scan.ranges = [float('inf')] * 10
            scan_pub.publish(scan)
        if state['started'] is not None:
            state['sequence'] += 1
            timing = DiagnosticArray(); timing.header.stamp = stamp
            item = DiagnosticStatus(); item.name = state['controller']
            item.values = [KeyValue(key='sequence', value=str(state['sequence'])),
                           KeyValue(key='duration_ns', value='1000'), KeyValue(key='success', value='true')]
            timing.status = [item]; timing_pub.publish(timing)
        if state['mode'] != 'late_streams' or (state['started'] is not None and time.monotonic() - state['started'] >= .5):
            command = Twist()
            if turning: command.angular.z = .6
            raw_pub.publish(command); applied_pub.publish(command)

    timer = robot.create_timer(1/30, publish)

    def execute(handle):
        path = handle.request.path
        assert path.header.frame_id == 'map'
        assert handle.request.controller_id == state['controller']
        assert handle.request.goal_checker_id == 'general_goal_checker'
        # E1_lateral has a constant zero reference yaw and a separate offset start.
        assert all(abs(p.pose.orientation.z) < 1e-12 for p in path.poses)
        state['seen_yaw'] = True
        state['started'] = time.monotonic()
        state['path'] = path.poses
        for i in range(len(path.poses)):
            if handle.is_cancel_requested:
                handle.canceled()
                return FollowPath.Result()
            state['index'] = i
            time.sleep(.01)
        state['completion_pending'] = True
        time.sleep(.2)
        state['finished'] = time.monotonic()
        if state.get('abort'):
            handle.abort()
        else:
            handle.succeed()
        return FollowPath.Result()

    def accept_goal(request):
        if state['mode'] == 'late_acceptance':
            time.sleep(10.5)
        return GoalResponse.ACCEPT

    server = ActionServer(nodes[0], FollowPath, '/follow_path', execute_callback=execute,
                          goal_callback=accept_goal, cancel_callback=lambda goal: CancelResponse.ACCEPT)
    executor = MultiThreadedExecutor(num_threads=4)
    for node in nodes: executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True); thread.start()
    try:
        with (nullcontext(args.output) if args.output else tempfile.TemporaryDirectory()) as directory:
            # The same read-only observer is used for start capture and the live
            # batch watchdog. Exercise actual DDS/TF with synchronized-but-offset
            # stamps, and ensure larger offsets never authorize a trial.
            sys.path.insert(0, str(ROOT/'scripts'))
            from dwvp_access_runtime import Observer
            for mode in ('small_skew', 'future_clock', 'old_clock'):
                state.update(path=None, started=None, mode=mode)
                observer = Observer()
                try:
                    if mode == 'small_skew':
                        captured = observer.stopped(timeout=3.)
                        assert captured['maximum_future_source_skew_s'] == .05
                        np.testing.assert_allclose(captured['pose'], [0., .5, 0.])
                    else:
                        try:
                            observer.stopped(timeout=2.)
                        except RuntimeError as exc:
                            assert 'Fresh map TF required' in str(exc), str(exc)
                        else:
                            raise AssertionError('Large clock offset must reject start capture')
                    assert not state['seen_yaw'], 'Start capture must not send a goal'
                    print(f'Start capture timing check passed: {mode}', flush=True)
                finally:
                    observer.destroy_node()
            for mode in ('noise_stationary', 'noise_moving', 'small_skew'):
                state.update(path=None, started=None, mode=mode)
                destination = Path(directory)/mode
                checked = subprocess.run([sys.executable,str(ROOT/'scripts/dwvp_access_experiment.py'),
                    'stationary-noise','--output',str(destination),'--duration','2.'],
                    timeout=10,capture_output=True,text=True)
                assert checked.returncode == (1 if mode=='noise_moving' else 0), checked.stdout+checked.stderr
                noise = json.loads((destination/'noise_summary.json').read_text())
                assert noise['stationary_verified'] == (mode!='noise_moving')
                if mode!='noise_moving':
                    assert noise['samples']>10 and noise['position_std_m']==0. and noise['heading_std_deg']==0., noise
                else:
                    assert noise['moving_samples']>10, noise
                assert not state['seen_yaw'], 'Noise observation must not send a goal'
                print(f'Stationary noise check passed: {mode}', flush=True)
            cases = [('normal', controller) for controller in experiment.default_config()['conditions']['E1_lateral']['methods']]
            cases += [(mode, 'DWVP') for mode in ('small_skew', 'delayed_final_pose', 'settled_outside', 'wrong_start', 'late_streams', 'stale_source', 'timeout', 'late_acceptance', 'interrupt',
                         'missing_scan', 'stale_scan', 'bad_scan_frame')]
            if args.clock_only:
                cases = [('small_skew', 'DWVP')]
            for mode, controller in cases:
                trial_id=f'E1_lateral_{controller}_r1'
                state.update(controller=controller, sequence=0, path=None, index=0, started=None, seen_yaw=False,
                             mode=mode, finished=None, completion_pending=False)
                # Exercise a 35 ms lead above the old 20 ms allowance. Faster
                # publication guarantees negative source ages are recorded.
                timer.timer_period_ns = int(1e9 / (200 if mode == 'small_skew' else 30))
                session = Path(directory) / (mode+'_'+controller)
                manifest = experiment.prepare(session, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0., 0., 0.])
                if mode in ('normal', 'small_skew', 'wrong_start', 'missing_scan', 'stale_scan', 'bad_scan_frame'):
                    checked = subprocess.run([sys.executable, str(ROOT / 'scripts/dwvp_access_experiment.py'), 'preflight',
                        '--session', str(session), '--trial', trial_id], timeout=40, capture_output=True, text=True)
                    assert not (session / 'runs').exists(), 'Preflight must not reserve a trial'
                    assert not state['seen_yaw'], 'Preflight must not send a motion goal'
                    if mode not in ('normal', 'small_skew'):
                        assert checked.returncode != 0 and ('frozen condition start pose' if mode == 'wrong_start' else 'Laser preflight failed') in checked.stderr, checked.stdout + checked.stderr
                        print(f'Synthetic ROS preflight rejection passed: {mode}', flush=True)
                        continue
                    assert checked.returncode == 0 and '"motion_goal_sent": false' in checked.stdout, checked.stdout + checked.stderr
                    previews.clear()  # wait for this case's message, not one left from an earlier case
                    preview = subprocess.Popen([sys.executable, str(ROOT / 'scripts/dwvp_access_experiment.py'), 'preview',
                        '--session', str(session), '--trial', trial_id], stdout=subprocess.DEVNULL)
                    try:
                        deadline = time.monotonic() + 5
                        while not previews and time.monotonic() < deadline: time.sleep(.02)
                        assert previews and len(previews[-1].poses) == 251
                        assert previews[-1].header.frame_id == 'map'
                        assert all(abs(p.pose.orientation.z) < 1e-12 for p in previews[-1].poses)
                        assert not state['seen_yaw'], 'Preview must not send a motion goal'
                    finally:
                        preview.send_signal(signal.SIGINT)
                        preview.wait(timeout=5)
                        assert preview.returncode == 0, 'Preview must close cleanly on Ctrl-C'
                if mode == 'timeout':
                    manifest['timeout_s'] = .2
                    experiment.write_json(session / 'manifest.json', manifest)
                command = [sys.executable, str(ROOT / 'scripts/dwvp_access_experiment.py'), 'run', '--session', str(session), '--trial', trial_id]
                if mode == 'interrupt':
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    deadline = time.monotonic() + 20
                    while state['started'] is None and time.monotonic() < deadline: time.sleep(.01)
                    assert state['started'] is not None
                    time.sleep(.1)
                    process.send_signal(signal.SIGINT)
                    stdout, stderr = process.communicate(timeout=10)
                    assert process.returncode != 0
                else:
                    completed = subprocess.run(command, timeout=45, capture_output=True, text=True)
                    assert completed.returncode == (1 if mode == 'late_acceptance' else 0), completed.stdout + completed.stderr
                report = experiment.summarize(session)
                assert report['recorded'] == 1 and report['pending'] == 79
                trial = report['trials'][0]
                result = json.loads((session / 'runs' / trial_id / 'result.json').read_text())
                if mode in ('normal', 'small_skew', 'late_streams', 'stale_source'):
                    assert trial['success'] == (mode != 'stale_source') and trial['valid_pose_samples'] > 30
                    if mode != 'stale_source':
                        # Preflight fixes the offset start. Recording may begin
                        # before or after the synthetic robot jumps onto the path.
                        np.testing.assert_allclose(result['preflight']['map_pose'], [0., .5, 0.])
                        assert trial['yaw_max_deg'] < 1e-8 and 0 <= trial['position_max_m'] <= .5 + 1e-6, (mode, trial_id, trial['yaw_max_deg'], trial['position_max_m'])
                    else:
                        # Preflight requires fresh odometry. The first recording tick
                        # can still use it before the first deliberately stale update.
                        tracking = np.genfromtxt(session / 'runs' / trial_id / 'tracking.csv',
                                                 delimiter=',', names=True)
                        max_age = experiment.default_config()['metrics']['maximum_source_age_s']
                        fresh = np.isfinite(np.column_stack(
                            [tracking[k] for k in ('x', 'y', 'yaw')])).all(axis=1)
                        for key in ('tf_age_s', 'odom_age_s', 'odom_source_age_s'):
                            fresh &= (tracking[key] >= 0) & (tracking[key] <= max_age)
                        stale = tracking['odom_source_age_s'] > max_age
                        assert stale.sum() > 30, 'The test must actually deliver stale odometry'
                        assert trial['fresh_pose_samples'] == int(fresh.sum()), trial
                        assert np.all(tracking['t'][fresh] <= max_age), 'Only the preflight prefix can be fresh'
                        assert trial['stale_or_missing_samples'] >= int(stale.sum()), trial
                    assert state['seen_yaw']
                if mode == 'small_skew':
                    tracking = np.genfromtxt(session/'runs'/trial_id/'tracking.csv', delimiter=',', names=True)
                    for key in ('tf_age_s', 'odom_source_age_s', 'scan_source_age_s'):
                        assert np.any(tracking[key] < 0), f'Test must actually record future stamps: {key}'
                    assert trial['fresh_pose_samples'] == trial['valid_pose_samples']
                if mode == 'late_streams':
                    assert trial['missing_command_prefix_s'] >= .45
                    assert trial['stale_or_missing_samples'] >= 10
                    assert next(g for g in report['groups'] if g['task']=='E1_lateral' and g['controller']==controller)['valid_successes'] == 0
                if mode == 'stale_source':
                    assert result['action_status'] == 4 and not result['settling']['verified']
                    assert trial['stale_or_missing_samples'] >= 10
                    assert next(g for g in report['groups'] if g['task']=='E1_lateral' and g['controller']==controller)['valid_successes'] == 0
                if mode in ('delayed_final_pose', 'settled_outside'):
                    assert result['action_status'] == 4 and result['settling']['verified']
                    assert result['success'] == (mode=='delayed_final_pose')
                    assert result['settling']['duration_s'] >= .5
                    samples = np.genfromtxt(session/'runs'/trial_id/'settling.csv', delimiter=',', names=True)
                    assert np.any(samples['yaw'] > .3), 'Must observe the delayed, outside-tolerance pose'
                    assert result['final_yaw_error_rad'] == (0.4 if mode=='settled_outside' else .02) or np.isclose(
                        result['final_yaw_error_rad'], .4 if mode=='settled_outside' else .02)
                if mode in ('timeout', 'late_acceptance', 'interrupt'):
                    assert not trial['success']
                    assert result['cancellation_confirmed'], result
                    assert result['cancellation_terminal_status'] == 5, result  # STATUS_CANCELED
                assert all((session / 'runs' / trial_id / f'{name}.csv').exists() for name in ('raw', 'applied', 'odom'))
                print(f'Synthetic ROS recorder case passed: {mode}', flush=True)
            # Only unscored positioning immediately followed by a new reference
            # may accept residual error. Action failures and stale inputs still fail.
            for label, mode, policy, abort, expected in (
                ('reanchor_residual', 'settled_outside', 'reanchor_after_stop', False, True),
                ('fixed_residual', 'settled_outside', 'fixed_tolerance', False, False),
                ('reanchor_stale', 'stale_source', 'reanchor_after_stop', False, False),
                ('reanchor_aborted', 'normal', 'reanchor_after_stop', True, False),
            ):
                state.update(controller='DWVP', sequence=0, path=None, index=0, started=None, seen_yaw=False,
                             mode=mode, finished=None, completion_pending=False, position_residual=.1013, abort=abort)
                session = Path(directory)/label
                manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml', [0,0,0])
                trial_id = 'E1_lateral_DWVP_r1'
                _, _, reference = experiment.load_trial(session, trial_id)
                folder = session/'alignment'
                packet = session/'transfer.json'
                experiment.write_json(packet, dict(path=reference.tolist(), start=[0,.5,0],
                    output=str(folder), endpoint_policy=policy))
                completed = subprocess.run([sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'), '_transfer',
                    '--session', str(session), '--trial', trial_id, '--packet', str(packet)],
                    timeout=45, capture_output=True, text=True)
                assert completed.returncode == (0 if expected else 1), completed.stdout+completed.stderr
                result = json.loads((folder/'result.json').read_text())
                assert result['success'] is expected and result['endpoint_policy'] == policy
                assert not (session/'runs').exists()
                if mode=='settled_outside':
                    assert result['settling']['verified'] and not result['final_pose_within_tolerances']
                    np.testing.assert_allclose(result['final_position_error_m'], .1013)
                    np.testing.assert_allclose(result['final_yaw_error_rad'], .4)
                elif mode=='stale_source':
                    assert not result['settling']['verified']
                else:
                    assert result['action_status'] == 6  # STATUS_ABORTED
                print(f'Synthetic transfer endpoint check passed: {label}', flush=True)
    finally:
        executor.shutdown()
        thread.join(timeout=2)
        for node in nodes: node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
