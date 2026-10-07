"""Read-only ROS observations shared by start capture and the batch supervisor."""
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


class Observer(Node):
    def __init__(self, base_frame='base_link', odom_topic='/omni_base_controller/wheel_odom'):
        super().__init__('dwvp_access_supervisor')
        self.base_frame = base_frame
        self.odom = self.scan = self.map = None
        self.applied = None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.subscriptions_owned = [
            self.create_subscription(Odometry, odom_topic, self.receive_odom, qos_profile_sensor_data),
            self.create_subscription(LaserScan, '/scan', self.receive_scan, qos_profile_sensor_data),
            self.create_subscription(OccupancyGrid, '/map', self.receive_map,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            self.create_subscription(Twist, '/omni_base_controller/cmd_vel', self.receive_applied, 10),
        ]

    def receive_odom(self, msg):
        self.odom = (msg, time.monotonic())

    def receive_scan(self, msg):
        self.scan = (msg, time.monotonic())

    def receive_map(self, msg):
        self.map = msg

    def receive_applied(self, msg):
        self.applied = (msg, time.monotonic())

    def await_zero_output(self, timeout):
        """Keep the smoother alive after its input stops, including sensor loss."""
        begin = time.monotonic()
        while time.monotonic() - begin < timeout:
            rclpy.spin_once(self, timeout_sec=.02)
            # Drain queued samples before inspecting the last command. The
            # smoother may cease publishing once it has reached exact zero.
            if time.monotonic() - begin < .6:
                continue
            if self.applied is None:
                return {'zero_output_observed': False, 'no_output_received': True}
            msg, _ = self.applied
            if max(abs(msg.linear.x), abs(msg.linear.y), abs(msg.angular.z)) < 1e-6:
                return {'zero_output_observed': True}
        raise RuntimeError('Smoother zero output not confirmed before shutdown')

    def state(self, require_scan=True):
        from dwvp_access_experiment import CLOCK_FUTURE_TOLERANCE_S, scan_quality, source_age_is_fresh
        tf = self.buffer.lookup_transform('map', self.base_frame, rclpy.time.Time())
        now = self.get_clock().now().nanoseconds / 1e9
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec / 1e9
        if not source_age_is_fresh(now - stamp):
            raise RuntimeError(f'Fresh map TF required (age {now-stamp:.3f} s; '
                               f'allowed {-CLOCK_FUTURE_TOLERANCE_S:.3f}..0.200 s); '
                               'check clock synchronization on both PC and robot')
        if self.odom is None:
            raise RuntimeError('Wheel odometry unavailable')
        odom, received = self.odom
        odom_age = now - (odom.header.stamp.sec + odom.header.stamp.nanosec / 1e9)
        if not source_age_is_fresh(odom_age) or not 0 <= time.monotonic() - received <= .2:
            raise RuntimeError(f'Fresh wheel odometry required (source age {odom_age:.3f} s)')
        if require_scan:
            if self.scan is None:
                raise RuntimeError('Laser scan unavailable')
            scan, received_scan = self.scan
            scan_quality(scan, now - (time.monotonic() - received_scan), now)
            self.buffer.lookup_transform(self.base_frame, scan.header.frame_id,
                                         rclpy.time.Time.from_msg(scan.header.stamp))
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        twist = odom.twist.twist
        velocities = [twist.linear.x, twist.linear.y, twist.angular.z]
        pose = [tf.transform.translation.x, tf.transform.translation.y, yaw]
        if not np.isfinite([*pose, *velocities]).all():
            raise RuntimeError('Non-finite pose/odometry')
        return dict(pose=pose, tf_stamp_s=stamp, captured_stamp_s=now, tf_age_s=now-stamp,
                    maximum_future_source_skew_s=CLOCK_FUTURE_TOLERANCE_S,
                    odom_source_age_s=odom_age, velocity=velocities,
                    stationary=math.hypot(*velocities[:2]) <= .005 and abs(velocities[2]) <= .01)

    def stopped(self, timeout=15., require_scan=True):
        deadline = time.monotonic() + timeout
        stable_since = None
        reason = 'No stationary localization received'
        while time.monotonic() < deadline:
            if getattr(self, 'cancel_requested', False):
                raise KeyboardInterrupt('Batch interrupted')
            rclpy.spin_once(self, timeout_sec=.02)
            try:
                state = self.state(require_scan)
                if not state['stationary']:
                    raise RuntimeError('Stop the robot before capturing a start or changing conditions')
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= .5:
                    return state
            except Exception as exc:
                reason = str(exc)
                stable_since = None
        raise RuntimeError(f'Stationary localized pose unavailable: {reason}')

    def command_owners(self, active):
        for topic, expected in (
                ('/omni_base_controller/cmd_vel', ['velocity_smoother'] if active else []),
                ('/cmd_vel_nav', ['controller_server'] if active else [])):
            publishers = self.get_publishers_info_by_topic(topic)
            names = sorted((p.node_namespace.rstrip('/') + '/' + p.node_name) for p in publishers)
            if names != ['/' + name for name in expected]:
                raise RuntimeError(f'Unexpected velocity publishers on {topic}: {names}; expected {expected}')


def capture_start(base_frame='base_link', odom_topic='/omni_base_controller/wheel_odom'):
    rclpy.init()
    observer = Observer(base_frame, odom_topic)
    try:
        # Capture is read-only and also usable before the controller is started.
        return observer.stopped(require_scan=False)
    finally:
        observer.destroy_node()
        rclpy.try_shutdown()
