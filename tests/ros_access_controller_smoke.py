"""Exercise real Humble plugins against a synthetic kinematic plant.

Run inside Docker with --network none and ROS_LOCALHOST_ONLY=1, after sourcing
Humble and the built DWPP/DWVP workspace. No hardware, AMCL, or TMC node is used.
This checks integration, not controller performance or physical-robot safety.
"""

import argparse
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


class Plant(Node):
    """Integrate the common smoother output; publish only synthetic sensors."""

    def __init__(self):
        super().__init__('access_controller_smoke_plant')
        self.x = self.y = self.yaw = 0.0
        self.applied = Twist()
        self.raw_count = self.applied_count = 0
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
        assert all(math.isfinite(v) for v in (
            command.linear.x, command.linear.y, command.angular.z))

    def applied_callback(self, command):
        self.applied_count += 1
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

    def reset(self):
        assert abs(self.applied.linear.x) + abs(self.applied.linear.y) + abs(
            self.applied.angular.z) < 1.0e-6, 'Smoother must stop before reset'
        self.x = self.y = self.yaw = 0.0
        self.raw_count = self.applied_count = 0
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


def run_case(plant, action, controller, independent_yaw):
    spin_for(plant, 1.3)
    plant.reset()
    spin_for(plant, 0.4)
    goal = FollowPath.Goal()
    goal.controller_id = controller
    goal.goal_checker_id = 'general_goal_checker'
    goal.path = RosPath()
    goal.path.header.frame_id = 'map'
    goal.path.header.stamp = plant.get_clock().now().to_msg()
    length = 0.9 if independent_yaw else 0.45
    for i in range(91):
        pose = PoseStamped()
        pose.header = goal.path.header
        pose.pose.position.x = length * i / 90.0
        # The independent task has the same XY line, but an explicit yaw ramp.
        yaw = 0.7 * min(1.0, i / 25.0) if independent_yaw else 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.path.poses.append(pose)
    future = action.send_goal_async(goal)
    spin_until(plant, future.done, 10.0)
    handle = future.result()
    assert handle.accepted, f'{controller} rejected path'
    result_future = handle.get_result_async()
    try:
        spin_until(plant, result_future.done, 35.0)
    except AssertionError:
        cancellation = handle.cancel_goal_async()
        spin_until(plant, cancellation.done, 5.0)
        raise
    result = result_future.result()
    report = {
        'controller': controller,
        'task': 'independent_yaw' if independent_yaw else 'tangent',
        'status': result.status,
        'final_x_m': plant.x,
        'final_y_m': plant.y,
        'final_yaw_rad': plant.yaw,
        'max_abs_yaw_rad': plant.max_yaw,
        'yaw_before_x_0_45_rad': plant.early_yaw,
        'max_abs_applied_vy_m_s': plant.max_vy,
        'raw_samples': plant.raw_count,
        'applied_samples': plant.applied_count,
    }
    print(json.dumps(report), flush=True)
    assert result.status == GoalStatus.STATUS_SUCCEEDED, report
    assert plant.raw_count > 10 and plant.applied_count > 10, report
    assert math.hypot(plant.x - length, plant.y) <= 0.12, report
    yaw_goal = 0.7 if independent_yaw else 0.0
    yaw_error = math.atan2(math.sin(plant.yaw - yaw_goal), math.cos(plant.yaw - yaw_goal))
    assert abs(yaw_error) <= 0.32, report
    if independent_yaw:
        assert plant.early_yaw > 0.1, report
        assert plant.max_vy > 0.01, report
    return report


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
    processes, logs = [], []
    rclpy.init()
    plant = Plant()
    try:
        for package, executable, remaps in (
            ('nav2_controller', 'controller_server', ['cmd_vel:=/cmd_vel_nav']),
            ('nav2_velocity_smoother', 'velocity_smoother', [
                'cmd_vel:=/cmd_vel_nav', 'cmd_vel_smoothed:=/omni_base_controller/cmd_vel']),
        ):
            log = (output / f'{executable}.log').open('w')
            logs.append(log)
            command = [f'/opt/ros/humble/lib/{package}/{executable}',
                       '--ros-args', '--params-file', str(args.params)]
            for remap in remaps:
                command.extend(['-r', remap])
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for name in ('controller_server', 'velocity_smoother'):
            transition(plant, name, Transition.TRANSITION_CONFIGURE)
            transition(plant, name, Transition.TRANSITION_ACTIVATE)
            state = service(plant, GetState, f'/{name}/get_state', GetState.Request())
            assert state.current_state.id == State.PRIMARY_STATE_ACTIVE
        request = GetParameters.Request()
        request.names = ['controller_plugins', 'MPPI.PathAlignCritic.use_path_orientations']
        response = service(plant, GetParameters, '/controller_server/get_parameters', request)
        assert list(response.values[0].string_array_value) == ['RPP', 'DWPP', 'MPPI', 'DWVP']
        assert response.values[1].bool_value
        print('All four plugins configured; controller and smoother active', flush=True)
        action = ActionClient(plant, FollowPath, '/follow_path')
        spin_until(plant, action.server_is_ready, 10.0)
        reports = [run_case(plant, action, name, False)
                   for name in ('RPP', 'DWPP', 'MPPI', 'DWVP')]
        reports.extend(run_case(plant, action, name, True) for name in ('MPPI', 'DWVP'))
        (output / 'report.json').write_text(json.dumps({
            'purpose': 'Synthetic integration smoke; not paper performance data',
            'params': str(args.params), 'cases': reports,
        }, indent=2) + '\n')
        print('PASS: four real controllers and two independent-yaw cases', flush=True)
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
        plant.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
