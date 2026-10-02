"""Synthetic ROS I/O test. Run only in an isolated container with --network none.

This is a recorder/action test, not evidence about robot/controller performance.
"""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import TransformBroadcaster

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def main():
    config = yaml.safe_load((ROOT / 'params/hsrb_dwvp_access_params.yaml').read_text())
    rclpy.init()
    nodes = []
    for name in ('controller_server', 'velocity_smoother'):
        overrides = [Parameter(k, value=v) for k, v in experiment.flatten_parameters(config[name]['ros__parameters']).items()]
        nodes.append(Node(name, parameter_overrides=overrides, automatically_declare_parameters_from_overrides=True))
    robot = Node('synthetic_robot')
    nodes.append(robot)
    broadcaster = TransformBroadcaster(robot)
    odom_pub = robot.create_publisher(Odometry, '/omni_base_controller/wheel_odom', 10)
    raw_pub = robot.create_publisher(Twist, '/cmd_vel_nav', 10)
    applied_pub = robot.create_publisher(Twist, '/omni_base_controller/cmd_vel', 10)
    state = {'path': None, 'index': 0, 'seen_yaw': False, 'mode': 'normal', 'started': None}

    def publish():
        if state['path'] is None:
            x, y, yaw = 0., 0., 0.
        else:
            p = state['path'][state['index']].pose
            x, y = p.position.x, p.position.y
            yaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
        stamp = robot.get_clock().now().to_msg()
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
        odom_pub.publish(odom)
        if state['mode'] != 'late_streams' or (state['started'] is not None and time.monotonic() - state['started'] >= .5):
            raw_pub.publish(Twist()); applied_pub.publish(Twist())

    timer = robot.create_timer(1/30, publish)

    def execute(handle):
        path = handle.request.path
        assert path.header.frame_id == 'map'
        assert handle.request.controller_id == 'DWVP'
        assert handle.request.goal_checker_id == 'general_goal_checker'
        # B_path1 must retain independent constant yaw despite turning XY.
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
        time.sleep(.2)
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
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('normal', 'late_streams', 'stale_source', 'timeout', 'late_acceptance'):
                state.update(path=None, index=0, started=None, seen_yaw=False, mode=mode)
                session = Path(directory) / mode
                manifest = experiment.prepare(session, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0., 0., 0.])
                if mode == 'timeout':
                    manifest['timeout_s'] = .2
                    experiment.write_json(session / 'manifest.json', manifest)
                completed = subprocess.run([sys.executable, str(ROOT / 'scripts/dwvp_access_experiment.py'), 'run', '--session', str(session), '--trial', 'B_path1_DWVP_r1'], timeout=45, capture_output=True, text=True)
                assert completed.returncode == (1 if mode == 'late_acceptance' else 0), completed.stdout + completed.stderr
                report = experiment.summarize(session)
                assert report['recorded'] == 1 and report['pending'] == 79
                trial = report['trials'][0]
                result = json.loads((session / 'runs/B_path1_DWVP_r1/result.json').read_text())
                if mode in ('normal', 'late_streams', 'stale_source'):
                    assert trial['success'] and trial['valid_pose_samples'] > 30
                    assert trial['yaw_max_deg'] < 1e-8 and trial['position_max_m'] < 1e-8
                    assert state['seen_yaw']
                if mode == 'late_streams':
                    assert trial['missing_command_prefix_s'] >= .45
                    assert trial['stale_or_missing_samples'] >= 10
                    assert report['groups'][0]['valid_successes'] == 0
                if mode == 'stale_source':
                    assert trial['stale_or_missing_samples'] >= 10
                    assert report['groups'][0]['valid_successes'] == 0
                if mode in ('timeout', 'late_acceptance'):
                    assert not trial['success']
                    assert result['cancellation_confirmed'], result
                    assert result['cancellation_terminal_status'] == 5, result  # STATUS_CANCELED
                assert all((session / 'runs/B_path1_DWVP_r1' / f'{name}.csv').exists() for name in ('raw', 'applied', 'odom'))
                print(f'Synthetic ROS recorder case passed: {mode}', flush=True)
    finally:
        executor.shutdown()
        thread.join(timeout=2)
        for node in nodes: node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
