#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import csv
import glob
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Twist
from nav_msgs.msg import Path, Odometry
from nav2_msgs.action import FollowPath, Spin
from visualization_msgs.msg import Marker, MarkerArray

from ros_gz_interfaces.srv import SetEntityPose  # Gazebo Sim (Ignition) の set_pose サービス
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency check
    yaml = None


def yaw_to_quat(z_yaw_rad: float):
    half = z_yaw_rad * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    return (0.0, 0.0, qz, qw)


def append_heading_to_path(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty((0, 3), dtype=float)
    if len(path_xy) == 1:
        return np.array([[path_xy[0, 0], path_xy[0, 1], 0.0]], dtype=float)

    diffs = np.diff(path_xy, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])
    return np.c_[path_xy, headings]


def right_angle_polyline_curve(segment_length: float = 1.0, points_per_segment: int = 100) -> np.ndarray:
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    x1 = np.linspace(0.0, segment_length, points_per_segment + 1)
    y1 = np.zeros_like(x1)
    x2 = np.full(points_per_segment + 1, segment_length)
    y2 = np.linspace(0.0, segment_length, points_per_segment + 1)

    # Remove duplicate corner point
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    return append_heading_to_path(np.c_[x, y])


def right_angle_polyline_curve_last_segment_heading_minus_pi(
    segment_length: float = 1.0,
    points_per_segment: int = 100,
) -> np.ndarray:
    path = right_angle_polyline_curve(
        segment_length=segment_length,
        points_per_segment=points_per_segment,
    ).copy()
    path[-1, 2] = -np.pi
    return path


def _resample_xy_by_arclength(x: np.ndarray, y: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total_length = float(s[-1])

    if total_length <= 1e-12:
        return (
            np.linspace(float(x[0]), float(x[-1]), num_points),
            np.linspace(float(y[0]), float(y[-1]), num_points),
        )

    s_new = np.linspace(0.0, total_length, num_points)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new


def one_minus_cos_curve(
    amplitude: float = 1.0,
    length_x: float = 2.0,
    num_points: int = 501,
    cycles: float = 1.5,
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
    resample_arclength: bool = True,
) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if length_x <= 0.0:
        raise ValueError("length_x must be > 0")

    a = float(amplitude)
    l = float(length_x)
    k = 2.0 * math.pi * float(cycles) / l

    x = np.linspace(float(x0), float(x0) + l, num_points, dtype=float)
    u = x - float(x0)
    y = float(y0) + a * (1.0 - np.cos(k * u))

    if abs(theta0) > 0.0:
        c = math.cos(theta0)
        s = math.sin(theta0)
        x_shift = x - float(x0)
        y_shift = y - float(y0)
        x = float(x0) + c * x_shift - s * y_shift
        y = float(y0) + s * x_shift + c * y_shift

    if resample_arclength:
        x, y = _resample_xy_by_arclength(x, y, num_points)

    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)
    return np.c_[x, y, theta]


def path_array_to_msg(path_xyz: np.ndarray, frame_id: str) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    for x, y, yaw in path_xyz:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        _, _, qz, qw = yaw_to_quat(float(yaw))
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        path.poses.append(ps)
    return path


def make_path(frame_id: str):
    paths_dict = {}

    # Only benchmark paths are used in this GUI.
    path3_xyz = right_angle_polyline_curve_last_segment_heading_minus_pi(
        segment_length=1.0,
        points_per_segment=100,
    )
    paths_dict["path3_right_angle_90_last_heading_minus_pi"] = path_array_to_msg(path3_xyz, frame_id)

    path4_xyz = one_minus_cos_curve(
        amplitude=0.75,
        length_x=1.5,
        num_points=501,
        cycles=1.5,
        resample_arclength=True,
    )
    paths_dict["path4_one_minus_cos"] = path_array_to_msg(path4_xyz, frame_id)

    return paths_dict


class FollowPathClient(Node):
    def __init__(self, map_frame_id: str = "map", local_path_frame_id: str = "local_path"):
        super().__init__('follow_path_gui_client')
        self._client = ActionClient(self, FollowPath, '/follow_path')
        self._spin_client = ActionClient(self, Spin, '/spin')
        self._current_goal_handle = None
        self._current_spin_goal_handle = None
        self._map_frame_id = map_frame_id
        self._local_path_frame_id = local_path_frame_id
        self._frame_id = map_frame_id

        # Visualization tuning parameters
        self._ref_path_line_width = 0.03
        self._ref_path_arrow_stride = 30
        self._ref_path_arrow_len_idx = 8
        self._ref_path_arrow_shaft_diameter = 0.045
        self._ref_path_arrow_head_diameter = 0.11
        self._ref_path_arrow_head_length = 0.17
        self._traj_line_width = 0.03
        self._traj_arrow_shaft_diameter = 0.07
        self._traj_arrow_head_diameter = 0.14
        self._traj_arrow_head_length = 0.20
        self._traj_arrow_tail_back_idx = 6
        
        # parameters
        self.robot_model_name = self.declare_parameter('robot_model_name', "turtlebot3_waffle").value
        self.world_model_name = self.declare_parameter('world_model_name', "empty").value
        self.nav2_params_file = self.declare_parameter('nav2_params_file', "").value
        self.base_frame_id = self.declare_parameter('base_frame_id', "base_link").value
        self.odom_topic = self.declare_parameter('odom_topic', "/odom").value
        self.strict_goal_checker_id = self.declare_parameter(
            'strict_goal_checker_id', "general_goal_checker").value
        self.dwpp_goal_checker_id = self.declare_parameter(
            'dwpp_goal_checker_id', "dwpp_goal_checker").value
        self.dwpp_controller_id = self.declare_parameter('dwpp_controller_id', "DWPP").value
        self.enable_dwpp_terminal_spin = bool(self.declare_parameter(
            'enable_dwpp_terminal_spin', True).value)
        self.dwpp_spin_time_allowance_sec = float(self.declare_parameter(
            'dwpp_spin_time_allowance_sec', 20.0).value)
        self.dwpp_spin_skip_yaw_error_rad = float(self.declare_parameter(
            'dwpp_spin_skip_yaw_error_rad', 0.03).value)

        default_controller_ids = ['PP', 'APP', 'RPP', 'DWPP']
        raw_controller_ids = self.declare_parameter('controller_ids', []).value
        if isinstance(raw_controller_ids, str):
            self.controller_ids = [raw_controller_ids]
        else:
            self.controller_ids = list(raw_controller_ids)

        if len(self.controller_ids) == 0:
            self.controller_ids = self._load_controller_ids_from_nav2_params(self.nav2_params_file)
        if len(self.controller_ids) == 0:
            self.controller_ids = default_controller_ids

        self.get_logger().info(f"GUI controller_ids: {self.controller_ids}")

        # --- QoS（RVizに残るように TRANSIENT_LOCAL） ---
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Initial pose publisher (RViz "2D Pose Estimate")
        self._initpose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)

        # --- Visualization publishers ---
        self._label_pub  = self.create_publisher(MarkerArray, '/viz/path_labels', latched_qos)
        self._path_markers_pub = self.create_publisher(MarkerArray, '/viz/path_markers', latched_qos)

        # Robot trajectory (subscribe odom -> publish Visualization MarkerArray)
        self._traj_pub = self.create_publisher(MarkerArray, '/viz/robot_trajs', 10)
        # 手法ごとに点列を保持
        self._traj_points = {controller_id: [] for controller_id in self.controller_ids}
        self._active_traj = None          # 現在アクティブな手法名（send_path時に設定）
        self._traj_frame_id = self._map_frame_id
        self._traj_lock = threading.Lock()

        # 手法→色マップ（R,G,B）
        base_color_map = {
            'DWPP':   (0.0, 0.4, 1.0),  # blue
            'VPmin':  (0.0, 0.7, 0.2),  # green
            'VP_MIN': (0.0, 0.7, 0.2),  # green (alias)
            'VPmax':  (1.0, 1.0, 0.0),  # yellow
            'VP_MAX': (1.0, 1.0, 0.0),  # yellow (alias)
            'DWVP':   (1.0, 0.0, 0.0),  # red
        }
        default_palette = [
            (1.0, 0.0, 0.0),
            (0.0, 0.7, 0.2),
            (0.0, 0.4, 1.0),
            (0.8, 0.2, 0.8),
            (1.0, 0.55, 0.0),
            (0.0, 0.8, 0.8),
            (1.0, 1.0, 0.0),
            (0.9, 0.4, 0.4),
        ]
        self._traj_colors = {}
        for i, controller_id in enumerate(self.controller_ids):
            self._traj_colors[controller_id] = base_color_map.get(
                controller_id, default_palette[i % len(default_palette)]
            )

        # 走行中のみ記録するためのフラグ
        self._recording = False

        # /odom 購読（SensorData QoS）
        self.current_odom = None
        self._odom_sub = self.create_subscription(
            Odometry, str(self.odom_topic), self._on_odom, qos_profile_sensor_data)
        self._last_cmd_vel_nav = Twist()
        self._cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)

        # TF (map -> base_frame) for local-path to map conversion
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._start_origin = None  # (x, y, yaw) in map frame
        self._local_paths = make_path(self._local_path_frame_id)
        self._last_goal_controller_id = None
        self._last_goal_target_yaw_map = None

        self._dwpp_csv_directory, self._dwpp_csv_prefix = self._load_dwpp_csv_settings_from_nav2_params(
            self.nav2_params_file)
        self._dwpp_csv_files_before_follow = set()
        self._dwpp_spin_logging_active = False
        self._dwpp_spin_log_csv_path = None
        if self._dwpp_csv_directory and os.path.isdir(self._dwpp_csv_directory):
            self.get_logger().info(
                f"DWPP CSV detection enabled: dir='{self._dwpp_csv_directory}', "
                f"prefix='{self._dwpp_csv_prefix}', odom_topic='{self.odom_topic}'")
        else:
            self.get_logger().warn(
                f"DWPP CSV directory not found from nav2 params: '{self._dwpp_csv_directory}'. "
                f"SPIN append may be disabled. odom_topic='{self.odom_topic}'")

        # Gazebo warp clients
        self._gz_setpose_cli = None  # /world/<world>/set_pose 用

        # 起動直後：初期姿勢 & tb3 ワープ & パス定期描画
        threading.Thread(target=self._auto_publish_initial_pose, daemon=True).start()
        threading.Thread(target=self._auto_warp, daemon=True).start()
        threading.Thread(target=self._periodic_path_publish, daemon=True).start()

    def _load_controller_ids_from_nav2_params(self, params_file: str):
        if not params_file:
            return []
        if yaml is None:
            self.get_logger().warn(
                "PyYAML is not available. Cannot read controller_plugins from nav2_params_file.")
            return []
        if not os.path.isfile(params_file):
            self.get_logger().warn(
                f"nav2_params_file not found: {params_file}. Falling back to default controller IDs.")
            return []

        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(
                f"Failed to parse nav2_params_file '{params_file}': {e}. "
                "Falling back to default controller IDs.")
            return []

        def find_controller_plugins(obj):
            if isinstance(obj, dict):
                ros_params = obj.get('ros__parameters')
                if isinstance(ros_params, dict) and 'controller_plugins' in ros_params:
                    return ros_params.get('controller_plugins')
                for value in obj.values():
                    found = find_controller_plugins(value)
                    if found is not None:
                        return found
            return None

        plugins = find_controller_plugins(data)
        if plugins is None:
            self.get_logger().warn(
                f"No controller_plugins found in '{params_file}'. Falling back to default controller IDs.")
            return []

        if isinstance(plugins, str):
            controller_ids = [plugins]
        elif isinstance(plugins, (list, tuple)):
            controller_ids = [str(x) for x in plugins if str(x).strip()]
        else:
            controller_ids = []

        if len(controller_ids) == 0:
            self.get_logger().warn(
                f"controller_plugins in '{params_file}' is empty. Falling back to default controller IDs.")
            return []

        self.get_logger().info(
            f"Loaded controller_ids from nav2_params_file '{params_file}': {controller_ids}")
        return controller_ids

    def _load_dwpp_csv_settings_from_nav2_params(self, params_file: str) -> tuple[str, str]:
        default_prefix = "dwpp_nav2"
        if not params_file:
            return ("", default_prefix)
        if yaml is None or (not os.path.isfile(params_file)):
            return ("", default_prefix)

        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return ("", default_prefix)

        def find_dwpp_block(obj):
            if isinstance(obj, dict):
                ros_params = obj.get('ros__parameters')
                if isinstance(ros_params, dict):
                    dwpp_block = ros_params.get(self.dwpp_controller_id)
                    if isinstance(dwpp_block, dict):
                        return dwpp_block
                for value in obj.values():
                    found = find_dwpp_block(value)
                    if found is not None:
                        return found
            return None

        dwpp = find_dwpp_block(data)
        if not isinstance(dwpp, dict):
            return ("", default_prefix)

        directory = str(dwpp.get('csv_log_directory', "")).strip()
        prefix = str(dwpp.get('csv_filename_prefix', default_prefix)).strip() or default_prefix
        if directory:
            directory = os.path.abspath(os.path.expanduser(directory))
        return (directory, prefix)

    def get_goal_checker_id_for_controller(self, controller_id: str) -> str:
        if controller_id == self.dwpp_controller_id:
            return self.dwpp_goal_checker_id
        return self.strict_goal_checker_id

    def _list_dwpp_csv_files(self) -> set[str]:
        if not self._dwpp_csv_directory or not self._dwpp_csv_prefix:
            return set()
        pattern = os.path.join(self._dwpp_csv_directory, f"{self._dwpp_csv_prefix}_*.csv")
        return {os.path.abspath(p) for p in glob.glob(pattern)}

    def _resolve_dwpp_csv_for_spin_log(self) -> str | None:
        files_after = self._list_dwpp_csv_files()
        if not files_after:
            self.get_logger().warn(
                f"DWPP CSV resolve failed: no files matched "
                f"dir='{self._dwpp_csv_directory}', prefix='{self._dwpp_csv_prefix}'.")
            return None

        new_files = sorted(files_after - self._dwpp_csv_files_before_follow)
        candidates = new_files if len(new_files) > 0 else sorted(files_after)
        if len(candidates) == 0:
            return None
        return max(candidates, key=lambda p: os.path.getmtime(p))

    def _quat_to_yaw(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _path_goal_yaw(self, path_msg: Path) -> float | None:
        if not path_msg.poses:
            return None
        q = path_msg.poses[-1].pose.orientation
        return self._quat_to_yaw(q)

    def _get_robot_pose_in_map(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self._map_frame_id, self.base_frame_id, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().debug(
                f"TF lookup failed ({self._map_frame_id}->{self.base_frame_id}): {ex}")
            return None
        yaw = self._quat_to_yaw(tf.transform.rotation)
        return (float(tf.transform.translation.x), float(tf.transform.translation.y), float(yaw))

    def _set_start_origin_to_current_pose(self) -> bool:
        pose = self._get_robot_pose_in_map()
        if pose is None:
            return False
        self._start_origin = pose
        return True

    def _transform_local_path_to_map_path(self, local_path: Path, origin):
        x0, y0, yaw0 = origin
        c = math.cos(yaw0)
        s = math.sin(yaw0)
        map_path = Path()
        map_path.header.frame_id = self._map_frame_id

        for ps in local_path.poses:
            xl = float(ps.pose.position.x)
            yl = float(ps.pose.position.y)
            zl = float(ps.pose.position.z)
            xm = x0 + c * xl - s * yl
            ym = y0 + s * xl + c * yl

            yaw_local = self._quat_to_yaw(ps.pose.orientation)
            yaw_map = yaw0 + yaw_local
            _, _, qz, qw = yaw_to_quat(yaw_map)

            out = PoseStamped()
            out.header.frame_id = self._map_frame_id
            out.pose.position.x = xm
            out.pose.position.y = ym
            out.pose.position.z = zl
            out.pose.orientation.z = qz
            out.pose.orientation.w = qw
            map_path.poses.append(out)
        return map_path

    def _build_map_paths_from_origin(self, origin):
        return {
            name: self._transform_local_path_to_map_path(path, origin)
            for name, path in self._local_paths.items()
        }

    def _publish_reference_paths_from_origin(self, origin):
        self.publish_paths_and_labels(self._build_map_paths_from_origin(origin))

    # ===== RViz 可視化：パス & ラベル =====
    def publish_paths_and_labels(self, paths_dict: dict[str, Path]):
        # Path visualization markers with distinct colors
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        palette = [
            (1.0, 0.0, 0.0),   # red
            (0.0, 1.0, 0.0),   # green
            (0.0, 0.0, 1.0),   # blue
            (1.0, 0.55, 0.0),  # orange
            (1.0, 1.0, 0.0),   # yellow
            (0.0, 1.0, 1.0),   # cyan
            (1.0, 0.0, 1.0),   # magenta
        ]

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        marker_id = 0
        for idx, (name, path) in enumerate(paths_dict.items()):
            if not path.poses:
                continue

            color = palette[idx % len(palette)]
            line = Marker()
            line.header.frame_id = path.header.frame_id
            line.header.stamp = now
            line.ns = 'path_visualization'
            line.id = marker_id
            marker_id += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = self._ref_path_line_width
            line.color.r = color[0]
            line.color.g = color[1]
            line.color.b = color[2]
            line.color.a = 1.0
            for pose_stamped in path.poses:
                line.points.append(pose_stamped.pose.position)
            markers.markers.append(line)

            # Add direction arrows on reference path for better visibility
            if len(path.poses) >= 2:
                for i in range(0, len(path.poses) - 1, self._ref_path_arrow_stride):
                    j = min(i + self._ref_path_arrow_len_idx, len(path.poses) - 1)
                    start = path.poses[i].pose.position
                    end = path.poses[j].pose.position
                    dx = end.x - start.x
                    dy = end.y - start.y
                    if (dx * dx + dy * dy) < 1e-8:
                        continue

                    arrow = Marker()
                    arrow.header.frame_id = path.header.frame_id
                    arrow.header.stamp = now
                    arrow.ns = 'path_visualization'
                    arrow.id = marker_id
                    marker_id += 1
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    arrow.scale.x = self._ref_path_arrow_shaft_diameter
                    arrow.scale.y = self._ref_path_arrow_head_diameter
                    arrow.scale.z = self._ref_path_arrow_head_length
                    arrow.color.r = color[0]
                    arrow.color.g = color[1]
                    arrow.color.b = color[2]
                    arrow.color.a = 0.95
                    arrow.points = [
                        Point(x=float(start.x), y=float(start.y), z=float(start.z + 0.02)),
                        Point(x=float(end.x), y=float(end.y), z=float(end.z + 0.02)),
                    ]
                    markers.markers.append(arrow)

        self._path_markers_pub.publish(markers)

        # Labels (TEXT_VIEW_FACING)
        labels = MarkerArray()
        clear_labels = Marker()
        clear_labels.action = Marker.DELETEALL
        labels.markers.append(clear_labels)

        for mid, (name, path) in enumerate(paths_dict.items(), start=1):
            if not path.poses:
                continue
            m = Marker()
            m.header.frame_id = path.header.frame_id
            m.header.stamp = now
            m.ns = 'path_labels'
            m.id = mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = path.poses[-1].pose.position.x + 0.1
            m.pose.position.y = path.poses[-1].pose.position.y + 0.1 + 0.08 * (mid - 1)
            m.pose.position.z = 0.35
            m.scale.z = 0.2
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 1.0
            m.text = name
            labels.markers.append(m)
        self._label_pub.publish(labels)

    # ===== Robot trajectory =====
    def _on_odom(self, msg: Odometry):
        self.current_odom = msg

        if self._dwpp_spin_logging_active:
            self._append_dwpp_spin_log_row(msg)

        with self._traj_lock:
            # 走行中でなければ記録しない（Warp などはここで無視）
            if not self._recording:
                return

            # Draw trajectory in map frame so it aligns with map-based reference paths.
            map_pose = self._get_robot_pose_in_map()
            if map_pose is not None:
                current_pos = Point()
                current_pos.x = float(map_pose[0])
                current_pos.y = float(map_pose[1])
                current_pos.z = float(msg.pose.pose.position.z)
                desired_frame = self._map_frame_id
            else:
                current_pos = msg.pose.pose.position
                desired_frame = msg.header.frame_id if msg.header.frame_id else self._traj_frame_id

            if self._traj_frame_id != desired_frame:
                # フレーム変更に追従（全軌跡クリア）
                for k in self._traj_points:
                    self._traj_points[k] = []
                self._traj_frame_id = desired_frame

            # 追従手法が未選択なら何もしない
            if self._active_traj not in self._traj_points:
                return

            pts = self._traj_points[self._active_traj]

            # 間引き（最後の点から5cm以上動いたら追加）
            if not pts or self._distance_2d(pts[-1], current_pos) > 0.05:
                pts.append(current_pos)
                if len(pts) > 5000:
                    self._traj_points[self._active_traj] = pts[-2000:]

            # 4手法すべてをまとめて MarkerArray で出す
            marr = MarkerArray()
            now = msg.header.stamp
            mid = 0
            for name, points in self._traj_points.items():
                if len(points) < 2:
                    continue
                r, g, b = self._traj_colors[name]
                m = Marker()
                m.header.frame_id = self._traj_frame_id
                m.header.stamp = now
                m.ns = 'robot_trajectory'
                m.id = mid; mid += 1
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = self._traj_line_width
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95
                m.points = points.copy()
                marr.markers.append(m)

                # Add a larger heading arrow at the current trajectory end
                tail_idx = max(0, len(points) - 1 - self._traj_arrow_tail_back_idx)
                tail = points[tail_idx]
                head = points[-1]
                dx = head.x - tail.x
                dy = head.y - tail.y
                if (dx * dx + dy * dy) >= 1e-8:
                    a = Marker()
                    a.header.frame_id = self._traj_frame_id
                    a.header.stamp = now
                    a.ns = 'robot_trajectory'
                    a.id = 1000 + mid; mid += 1
                    a.type = Marker.ARROW
                    a.action = Marker.ADD
                    a.scale.x = self._traj_arrow_shaft_diameter
                    a.scale.y = self._traj_arrow_head_diameter
                    a.scale.z = self._traj_arrow_head_length
                    a.color.r = r; a.color.g = g; a.color.b = b; a.color.a = 1.0
                    a.points = [
                        Point(x=float(tail.x), y=float(tail.y), z=float(tail.z + 0.02)),
                        Point(x=float(head.x), y=float(head.y), z=float(head.z + 0.02)),
                    ]
                    marr.markers.append(a)

            self._traj_pub.publish(marr)

    def _on_cmd_vel(self, msg: Twist):
        self._last_cmd_vel_nav = msg

    def _append_dwpp_spin_log_row(self, odom_msg: Odometry):
        csv_path = self._dwpp_spin_log_csv_path
        if not csv_path:
            return

        stamp = odom_msg.header.stamp
        sec = int(stamp.sec)
        nsec = int(stamp.nanosec)
        if sec == 0 and nsec == 0:
            now = self.get_clock().now().to_msg()
            sec = int(now.sec)
            nsec = int(now.nanosec)

        yaw = self._quat_to_yaw(odom_msg.pose.pose.orientation)
        map_pose = self._get_robot_pose_in_map()
        if map_pose is None:
            map_x, map_y, map_yaw, map_pose_valid = float("nan"), float("nan"), float("nan"), 0
        else:
            map_x, map_y, map_yaw = map_pose
            map_pose_valid = 1

        cmd_v = float(self._last_cmd_vel_nav.linear.x)
        cmd_w = float(self._last_cmd_vel_nav.angular.z)

        row = [
            sec,
            nsec,
            float(odom_msg.pose.pose.position.x),
            float(odom_msg.pose.pose.position.y),
            float(yaw),
            float(map_x),
            float(map_y),
            float(map_yaw),
            int(map_pose_valid),
            float(odom_msg.twist.twist.linear.x),
            float(odom_msg.twist.twist.angular.z),
            cmd_v,   # v_now
            cmd_w,   # w_now
            cmd_v,   # v_cmd
            cmd_w,   # w_cmd
            cmd_v,   # v_nav
            cmd_w,   # w_nav
            0,       # velocity_violation
            float("nan"),  # curvature
            float("nan"),  # dw_v_max
            float("nan"),  # dw_v_min
            float("nan"),  # dw_w_max
            float("nan"),  # dw_w_min
            float("nan"),  # v_reg
        ]

        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            self.get_logger().warn(f"Failed to append DWPP spin log row to '{csv_path}': {e}")
            self._dwpp_spin_logging_active = False
            self._dwpp_spin_log_csv_path = None

    def _start_dwpp_terminal_spin(self):
        if not self.enable_dwpp_terminal_spin:
            return
        if self._last_goal_target_yaw_map is None:
            self.get_logger().warn("DWPP terminal spin skipped: target yaw is unavailable.")
            return

        if not self._spin_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("`/spin` action server not available. Skipping terminal spin.")
            return

        pose = self._get_robot_pose_in_map()
        if pose is None:
            self.get_logger().warn("DWPP terminal spin skipped: current pose in map is unavailable.")
            return

        current_yaw = pose[2]
        target_delta = self._normalize_angle(self._last_goal_target_yaw_map - current_yaw)
        if abs(target_delta) < self.dwpp_spin_skip_yaw_error_rad:
            self.get_logger().info(
                f"DWPP terminal spin skipped: yaw error {target_delta:.4f} rad is below threshold.")
            return

        spin_goal = Spin.Goal()
        spin_goal.target_yaw = float(target_delta)
        if hasattr(spin_goal, "time_allowance"):
            sec_int = int(self.dwpp_spin_time_allowance_sec)
            nsec_int = int((self.dwpp_spin_time_allowance_sec - sec_int) * 1e9)
            spin_goal.time_allowance = Duration(sec=sec_int, nanosec=nsec_int)
        if hasattr(spin_goal, "disable_collision_checks"):
            spin_goal.disable_collision_checks = True

        self._dwpp_spin_log_csv_path = self._resolve_dwpp_csv_for_spin_log()
        self._dwpp_spin_logging_active = self._dwpp_spin_log_csv_path is not None
        if self._dwpp_spin_logging_active:
            self.get_logger().info(f"Appending DWPP spin log to: {self._dwpp_spin_log_csv_path}")
        else:
            self.get_logger().warn("DWPP spin log append skipped: DWPP CSV file was not found.")

        send_future = self._spin_client.send_goal_async(spin_goal)

        def _spin_goal_response_cb(fut):
            self._current_spin_goal_handle = fut.result()
            if not self._current_spin_goal_handle.accepted:
                self.get_logger().warn("DWPP terminal spin goal rejected.")
                self._current_spin_goal_handle = None
                self._dwpp_spin_logging_active = False
                self._dwpp_spin_log_csv_path = None
                return
            self.get_logger().info(
                f"DWPP terminal spin accepted (target_delta={target_delta:.4f} rad).")
            self._current_spin_goal_handle.get_result_async().add_done_callback(self._spin_result_cb)

        send_future.add_done_callback(_spin_goal_response_cb)

    def _spin_result_cb(self, fut):
        self._current_spin_goal_handle = None
        self._dwpp_spin_logging_active = False
        self._dwpp_spin_log_csv_path = None
        try:
            status = fut.result().status
            self.get_logger().info(f"DWPP terminal spin finished. status={status}")
        except Exception as e:
            self.get_logger().warn(f"Spin result callback error: {e}")

    def _distance_2d(self, p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(dx*dx + dy*dy)

    def clear_trajectory(self):
        with self._traj_lock:
            for k in self._traj_points:
                self._traj_points[k] = []

        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self._traj_pub.publish(ma)

        self.get_logger().info(f"Cleared ALL robot trajectories ({'/'.join(self.controller_ids)}).")

    # ===== Initial pose =====
    def publish_initial_pose(self, x: float = 0.0, y: float = 0.0, yaw_rad: float = 0.0):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        _, _, qz, qw = yaw_to_quat(yaw_rad)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = 0.0
        cov[7] = 0.0
        cov[35] = 0.0
        msg.pose.covariance = cov

        self._initpose_pub.publish(msg)
        self.get_logger().info(f"Published initial pose at ({x:.2f}, {y:.2f}, yaw={yaw_rad:.2f} rad) in frame '{self._frame_id}'")

    def _auto_publish_initial_pose(self):
        time.sleep(0.5)
        for _ in range(3):
            self.publish_initial_pose(0.0, 0.0, 0.0)
            time.sleep(0.5)

    # ===== Gazebo warp =====
    def _ensure_gazebo_clients(self):
        if self._gz_setpose_cli is None:
            service_name = f'/world/{self.world_model_name}/set_pose'
            self._gz_setpose_cli = self.create_client(SetEntityPose, service_name)

    def warp_model(self, model_name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_rad: float = 0.0):
        # Warp の前に記録OFF（Warp移動は記録しない）
        with self._traj_lock:
            self._recording = False
        # The next follow run should set a fresh start-origin.
        self._start_origin = None

        self._ensure_gazebo_clients()
        _, _, qz, qw = yaw_to_quat(yaw_rad)
        if not self._gz_setpose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"'/world/{self.world_model_name}/set_pose' service not available. Is ros_gz_bridge running?")
            return False
        # SetEntityPose リクエスト（name 指定 / type=MODEL=2）
        req = SetEntityPose.Request()
        req.entity.name = model_name
        req.entity.type = 2  # MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw
        fut = self._gz_setpose_cli.call_async(req)
        def _done(f):
            try:
                _ = f.result()
                self.get_logger().info(
                    f"Warped '{model_name}' via /world/{self.world_model_name}/set_pose to ({x:.2f},{y:.2f},{z:.2f})")
            except Exception as e:
                self.get_logger().warn(f"set_pose failed: {e}")
        fut.add_done_callback(_done)
        return True

    def _auto_warp(self):
        time.sleep(1.5)
        try:
            # 起動直後に turtlebot3_waffle を原点へテレポート
            self.warp_model(self.robot_model_name, 0.0, 0.0, 0.0, 0.0)
        except Exception as e:
            self.get_logger().warn(f"Auto-warp failed: {e}")

    def _periodic_path_publish(self):
        time.sleep(2.0)  # 初期化待ち
        while rclpy.ok():
            try:
                origin = self._start_origin
                if origin is None:
                    origin = self._get_robot_pose_in_map()
                if origin is None:
                    time.sleep(1.0)
                    continue

                self._publish_reference_paths_from_origin(origin)
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().warn(f"Periodic path publish failed: {e}")
                time.sleep(5.0)

    # ===== follow_path action =====
    def send_path(self, local_path_msg: Path, controller_id: str, goal_checker_id: str):
        if not self._client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("`/follow_path` action server not available.")

        if not self._set_start_origin_to_current_pose():
            raise RuntimeError(
                f"Could not get current robot pose in '{self._map_frame_id}'. "
                "Check TF map->base_link and localization state.")
        # Rewrite reference paths in RViz with this run's start origin immediately.
        self._publish_reference_paths_from_origin(self._start_origin)
        path_msg = self._transform_local_path_to_map_path(local_path_msg, self._start_origin)
        self._last_goal_controller_id = controller_id
        self._last_goal_target_yaw_map = self._path_goal_yaw(path_msg)
        self._dwpp_spin_logging_active = False
        self._dwpp_spin_log_csv_path = None
        if controller_id == self.dwpp_controller_id:
            self._dwpp_csv_files_before_follow = self._list_dwpp_csv_files()
        else:
            self._dwpp_csv_files_before_follow = set()

        goal = FollowPath.Goal()
        goal.path = path_msg
        goal.controller_id = controller_id
        goal.goal_checker_id = goal_checker_id

        with self._traj_lock:
            if controller_id not in self._traj_points:
                self._traj_points[controller_id] = []
                self._traj_colors[controller_id] = (1.0, 1.0, 1.0)
            self._active_traj = controller_id
            # 手法切替時はその手法の軌跡をクリアして「新しい走行」として描く
            self._traj_points[controller_id] = []
            self._recording = True

        send_future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)

        def _goal_response_cb(fut):
            self._current_goal_handle = fut.result()
            if not self._current_goal_handle.accepted:
                self.get_logger().warn('Goal rejected by controller_server.')
                # 受理されなかったら記録OFFに戻す
                with self._traj_lock:
                    self._recording = False
                return
            self.get_logger().info(f'Goal accepted by controller "{controller_id}".')
            self._current_goal_handle.get_result_async().add_done_callback(self._result_cb)

        send_future.add_done_callback(_goal_response_cb)

    def cancel_current_goal(self):
        gh = self._current_goal_handle
        if gh is not None:
            cancel_future = gh.cancel_goal_async()
            cancel_future.add_done_callback(lambda _: self.get_logger().info('FollowPath cancel request sent.'))
        else:
            self.get_logger().info('No active FollowPath goal to cancel.')

        spin_gh = self._current_spin_goal_handle
        if spin_gh is not None:
            cancel_future = spin_gh.cancel_goal_async()
            cancel_future.add_done_callback(lambda _: self.get_logger().info('Spin cancel request sent.'))
            self._current_spin_goal_handle = None

        # キャンセルしたら記録OFF
        with self._traj_lock:
            self._recording = False
        self._dwpp_spin_logging_active = False
        self._dwpp_spin_log_csv_path = None

    def _feedback_cb(self, feedback_msg):
        self.get_logger().debug(f'Feedback: {feedback_msg}')

    def _result_cb(self, fut):
        # ゴール終了で必ず記録OFF
        with self._traj_lock:
            self._recording = False

        try:
            result = fut.result().result
            status = fut.result().status
            self.get_logger().info(f'Result received. status={status}, result={result}')
            if (
                status == GoalStatus.STATUS_SUCCEEDED
                and self._last_goal_controller_id == self.dwpp_controller_id
            ):
                self._start_dwpp_terminal_spin()
        except Exception as e:
            self.get_logger().warn(f'Result callback error: {e}')


class AppGUI:
    def __init__(self, node: FollowPathClient, paths_dict: dict, frame_id: str):
        self.node = node
        self.paths_dict = paths_dict
        self.frame_id = frame_id

        self.root = tk.Tk()
        self.root.title("FollowPath GUI (Nav2)")
        self.root.geometry("960x540")
        
        self.robot_model_name = self.node.robot_model_name
        self.world_model_name = self.node.world_model_name
        
        # Set larger default font for the whole application
        default_font = ("Arial", 20)
        self.root.option_add("*Font", default_font)

        tk.Label(self.root, text=f"path_frame: {self.frame_id} (start-relative)", font=("Arial", 20)).pack(pady=8)

        frm = tk.Frame(self.root); frm.pack(pady=8)

        tk.Label(frm, text="Controller:", font=("Arial", 20)).grid(row=0, column=0, sticky="e")
        default_controller = self.node.controller_ids[0] if self.node.controller_ids else "PP"
        self.controller_var = tk.StringVar(value=default_controller)
        self.controller_cb = ttk.Combobox(frm, textvariable=self.controller_var,
                                          values=self.node.controller_ids, state="readonly", width=10, font=("Arial", 20, "bold"))
        self.controller_cb.grid(row=0, column=1, padx=6)

        # Buttons row
        btn_row = tk.Frame(self.root); btn_row.pack(pady=(8, 12))
        tk.Button(btn_row, text="Warp Robot (0,0,0)", font=("Arial", 20, "bold"), command=self._on_warp).grid(row=0, column=0, padx=8)
        tk.Button(btn_row, text="Clear Trajectory", font=("Arial", 20, "bold"), command=self._on_clear_traj).grid(row=0, column=1, padx=8)

        tk.Label(self.root, text="Paths", font=("Arial", 20)).pack(pady=(10, 4))
        btns = tk.Frame(self.root); btns.pack()

        for i, name in enumerate(self.paths_dict.keys()):
            tk.Button(btns, text=name, width=34, font=("Arial", 14, "bold"),
                      command=lambda n=name: self._on_send(n)).grid(row=i, column=0, padx=8, pady=6, sticky="ew")

        tk.Button(self.root, text="Cancel", font=("Arial", 20, "bold"), command=self._on_cancel).pack(pady=(12, 8))

    def _on_set_initial_pose(self):
        try:
            self.node.publish_initial_pose(0.0, 0.0, 0.0)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_warp(self):
        try:
            ok = self.node.warp_model(self.robot_model_name, 0.0, 0.0, 0.0, 0.0)
            if not ok:
                messagebox.showwarning("Warp", f"Failed to warp {self.robot_model_name}. Check Gazebo services.")
        except Exception as e:
            messagebox.showerror("Warp Error", str(e))

    def _on_clear_traj(self):
        try:
            self.node.clear_trajectory()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_send(self, path_name: str):
        try:
            controller_id = self.controller_var.get()
            goal_checker_id = self.node.get_goal_checker_id_for_controller(controller_id)
            path_msg = self.paths_dict[path_name]
            self.node.get_logger().info(
                f"Sending path '{path_name}' using controller '{controller_id}' "
                f"with goal_checker '{goal_checker_id}'")
            self.node.send_path(path_msg, controller_id, goal_checker_id)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_cancel(self):
        self.node.cancel_current_goal()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()

    local_frame_id = "local_path"
    paths = make_path(local_frame_id)

    node = FollowPathClient(map_frame_id="map", local_path_frame_id=local_frame_id)
    # === 安全な executor / spin ===
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        gui = AppGUI(node, paths, local_frame_id)
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
