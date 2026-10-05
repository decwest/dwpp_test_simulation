"""Exercise real Humble plugins against a synthetic kinematic plant.

Run inside Docker with --network none and ROS_LOCALHOST_ONLY=1, after sourcing
Humble and the built DWPP/DWVP workspace. No hardware, AMCL, or TMC node is used.
This checks integration, not controller performance or physical-robot safety.
"""

import argparse
from contextlib import contextmanager
import importlib.util
import numpy as np
import yaml
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('experiment', ROOT/'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def synthetic_environment_path():
    x = np.linspace(0, 1.2, 121)
    xy = np.c_[x, .15*np.sin(np.pi*x/1.2)]
    return np.c_[xy, np.arctan2(np.gradient(xy[:,1]), np.gradient(xy[:,0]))]



class Plant(Node):
    """Integrate the common smoother output; publish only synthetic sensors."""

    def __init__(self):
        super().__init__('access_controller_smoke_plant')
        self.x = self.y = self.yaw = 0.0
        self.applied = Twist()
        self.raw_count = self.applied_count = 0
        self.raw_history = []
        self.applied_history = []
        self.max_yaw = self.max_vy = self.early_yaw = 0.0
        self.last_tick = time.monotonic()
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'odom'
        transform.transform.rotation.w = 1.0
        self.static_tf.sendTransform(transform)
        self.odom_pub = self.create_publisher(
            Odometry, '/omni_base_controller/wheel_odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Twist, '/cmd_vel_nav', self.raw_callback, 10)
        self.create_subscription(
            Twist, '/omni_base_controller/cmd_vel', self.applied_callback, 10)
        self.create_timer(1.0 / 100.0, self.tick)
        self.create_timer(0.1, self.publish_scan)
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = 0.05
        grid.info.width = grid.info.height = 200
        grid.info.origin.position.x = grid.info.origin.position.y = -5.0
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (200 * 200)
        self.map_pub.publish(grid)

    def raw_callback(self, command):
        self.raw_count += 1
        self.raw_history.append([command.linear.x, command.linear.y, command.angular.z])
        assert all(math.isfinite(v) for v in (
            command.linear.x, command.linear.y, command.angular.z))

    def applied_callback(self, command):
        self.applied_count += 1
        self.applied_history.append([command.linear.x, command.linear.y, command.angular.z])
        self.applied = command
        self.max_vy = max(self.max_vy, abs(command.linear.y))
        assert abs(command.linear.x) <= 0.220001
        assert abs(command.linear.y) <= 0.220001
        assert abs(command.angular.z) <= 0.600001

    def tick(self):
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        vx, vy, omega = (
            self.applied.linear.x, self.applied.linear.y, self.applied.angular.z)
        if abs(omega) > 1.0e-9:
            dx = (vx * math.sin(omega * dt) - vy * (1.0 - math.cos(omega * dt))) / omega
            dy = (vx * (1.0 - math.cos(omega * dt)) + vy * math.sin(omega * dt)) / omega
        else:
            dx, dy = vx * dt, vy * dt
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += c * dx - s * dy
        self.y += s * dx + c * dy
        self.yaw += omega * dt
        self.max_yaw = max(self.max_yaw, abs(self.yaw))
        if self.x < 0.45:
            self.early_yaw = max(self.early_yaw, abs(self.yaw))
        stamp = self.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.frame_id = 'odom'
        transform.header.stamp = stamp
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.rotation.z = math.sin(self.yaw / 2.0)
        transform.transform.rotation.w = math.cos(self.yaw / 2.0)
        self.tf.sendTransform(transform)
        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.header.stamp = stamp
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = transform.transform.rotation
        odom.twist.twist = self.applied
        self.odom_pub.publish(odom)

    def publish_scan(self):
        scan = LaserScan()
        scan.header.frame_id = 'base_link'
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = 2.0 * math.pi / 180.0
        scan.range_min = 0.05
        scan.range_max = 5.0
        # Beyond obstacle_max_range, within declared sensor range: free space.
        scan.ranges = [3.5] * 181
        self.scan_pub.publish(scan)

    def reset(self, pose=(0., 0., 0.)):
        assert abs(self.applied.linear.x) + abs(self.applied.linear.y) + abs(
            self.applied.angular.z) < 1.0e-6, 'Smoother must stop before reset'
        self.x, self.y, self.yaw = pose
        self.raw_count = self.applied_count = 0
        self.raw_history = []
        self.applied_history = []
        self.max_yaw = self.max_vy = self.early_yaw = 0.0


def spin_until(node, predicate, timeout):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'Timed out waiting for ROS operation'
        rclpy.spin_once(node, timeout_sec=0.02)


def spin_for(node, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)


def service(node, service_type, name, request):
    client = node.create_client(service_type, name)
    try:
        spin_until(node, client.service_is_ready, 15.0)
        future = client.call_async(request)
        spin_until(node, future.done, 25.0)
        return future.result()
    finally:
        node.destroy_client(client)


def transition(node, target, transition_id):
    request = ChangeState.Request()
    request.transition.id = transition_id
    response = service(node, ChangeState, f'/{target}/change_state', request)
    assert response.success, f'{target} rejected transition {transition_id}'


def run_case(plant, action, controller, condition):
    spin_for(plant, 2.3)
    config = experiment.default_config()
    reference = synthetic_environment_path() if condition == 'E2_environment' else experiment.canonical_path(condition, config)
    initial = reference[0] if condition == 'E2_environment' else config['conditions'][condition]['start_pose']
    plant.reset(initial)
    spin_for(plant, .4)
    goal = FollowPath.Goal()
    goal.controller_id = controller
    goal.goal_checker_id = 'general_goal_checker'
    goal.path = experiment.pose_path_message(reference, 'map', plant.get_clock().now().to_msg())
    future = action.send_goal_async(goal)
    spin_until(plant, future.done, 10.)
    handle = future.result()
    assert handle.accepted, f'{controller} rejected {condition}'
    result_future = handle.get_result_async()
    try:
        spin_until(plant, result_future.done, 120.)
    except AssertionError:
        cancellation = handle.cancel_goal_async()
        spin_until(plant, cancellation.done, 5.)
        raise
    result = result_future.result()
    position_error = math.hypot(plant.x-reference[-1,0], plant.y-reference[-1,1])
    yaw_error = float(experiment.wrap(plant.yaw-reference[-1,2]))
    report = {'controller':controller,'task':condition,'status':result.status,
              'final_position_error_m':position_error,'final_yaw_error_rad':yaw_error,
              'max_abs_applied_vy_m_s':plant.max_vy,'raw_samples':plant.raw_count,'applied_samples':plant.applied_count}
    print(json.dumps(report), flush=True)
    assert result.status == GoalStatus.STATUS_SUCCEEDED, report
    assert plant.raw_count > 10 and plant.applied_count > 10, report
    assert position_error <= .12 and abs(yaw_error) <= .32, report
    if condition == 'E1_orientation_half':
        report['half_acceleration'] = check_half_acceleration(plant)
    return report


def check_half_acceleration(plant):
    raw = np.asarray(plant.raw_history)
    # Nav2 publishes an extra terminal stop outside computeVelocityCommands.
    # Exclude only that zero suffix from the controller per-call increment test.
    while len(raw) and np.all(raw[-1] == 0):
        raw = raw[:-1]
    bound = np.array([.11, .11, .3]) / 30.
    assert len(raw) > 30
    changes = np.abs(np.diff(np.vstack((np.zeros(3), raw)), axis=0))
    maximum = changes.max(axis=0)
    assert np.all(maximum <= bound + 1e-8), (maximum, bound)
    applied = np.asarray(plant.applied_history)
    applied_max = np.abs(np.diff(np.vstack((np.zeros(3), applied)), axis=0)).max(axis=0)
    assert np.all(applied_max <= bound + 1e-8), (applied_max, bound)
    return dict(controller_max_step=maximum.tolist(), smoother_max_step=applied_max.tolist(),
                bound=bound.tolist(), controller_samples=len(raw),
                terminal_server_zero_excluded=True)


@contextmanager
def controller_stack(plant, params, output):
    """Start only isolated synthetic controller/smoother nodes; always reap them."""
    output.mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    try:
        for package, executable, remaps in (
            ('nav2_controller', 'controller_server', ['cmd_vel:=/cmd_vel_nav']),
            ('nav2_velocity_smoother', 'velocity_smoother', [
                'cmd_vel:=/cmd_vel_nav', 'cmd_vel_smoothed:=/omni_base_controller/cmd_vel']),
        ):
            log = (output / f'{executable}.log').open('w')
            logs.append(log)
            command = [f'/opt/ros/humble/lib/{package}/{executable}',
                       '--ros-args', '--params-file', str(params)]
            for remap in remaps:
                command.extend(['-r', remap])
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for name in ('controller_server', 'velocity_smoother'):
            transition(plant, name, Transition.TRANSITION_CONFIGURE)
            transition(plant, name, Transition.TRANSITION_ACTIVATE)
            state = service(plant, GetState, f'/{name}/get_state', GetState.Request())
            assert state.current_state.id == State.PRIMARY_STATE_ACTIVE
        yield
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        spin_for(plant, .5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=ROOT / 'params/hsrb_dwvp_access_params.yaml')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    assert Path('/.dockerenv').exists(), 'This test requires an isolated Docker container'
    assert set(os.listdir('/sys/class/net')) == {'lo'}, 'Use Docker --network none'
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1', 'Set ROS_LOCALHOST_ONLY=1'
    output = args.output or Path(tempfile.mkdtemp(prefix='access-controller-smoke-'))
    output.mkdir(parents=True, exist_ok=True)
    print(f'Actual Humble controller integration logs: {output}', flush=True)
    rclpy.init()
    plant = Plant()
    reports = []
    config = experiment.default_config()
    try:
        for profile, conditions in (
            ('nominal', ['E1_lateral', 'E1_orientation_nominal', 'E2_environment']),
            ('half', ['E1_orientation_half']),
        ):
            materialized = output / f'nav2_params_{profile}.yaml'
            materialized.write_text(yaml.safe_dump(experiment.render_parameters(args.params, config, conditions[0])))
            with controller_stack(plant, materialized, output / profile):
                request = GetParameters.Request()
                request.names = ['controller_plugins', 'MPPI.PathAlignCritic.use_path_orientations',
                                 'DWB.trajectory_generator_name', 'DWB.max_vel_y', 'VP_SCALED.use_uniform_velocity_scaling']
                response = service(plant, GetParameters, '/controller_server/get_parameters', request)
                assert list(response.values[0].string_array_value) == list(experiment.CONTROLLERS)
                assert response.values[1].bool_value and response.values[4].bool_value
                assert response.values[2].string_value == 'dwb_plugins::LimitedAccelGenerator'
                assert response.values[3].double_value == .22
                action = ActionClient(plant, FollowPath, '/follow_path')
                spin_until(plant, action.server_is_ready, 10.0)
                try:
                    for condition in conditions:
                        for name in config['conditions'][condition]['methods']:
                            reports.append(run_case(plant, action, name, condition))
                    spin_for(plant, 2.3)
                finally:
                    action.destroy()
        (output / 'report.json').write_text(json.dumps({
            'purpose': 'Synthetic integration smoke on uncommitted working tree; not paper performance data',
            'cases': reports, 'physical_trials': 0,
        }, indent=2) + '\n')
        print('PASS: seven controllers, four condition/profile combinations, half-acceleration bounds', flush=True)
    finally:
        plant.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
