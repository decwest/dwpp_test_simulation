#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nav2 FollowPath Test GUI Client (map統一 + ロボット原点ローカル経路版)

目的
- 参照経路は「追従開始時のロボット姿勢」を原点とするローカル座標で定義 (local_path)
- 送信時点のロボット map 座標姿勢を取得し、local_path を map に数値変換して FollowPath へ送る
- RViz へ描画する経路も、Nav2 に送った map_path と完全に同一のものを描画する
- start_pose TF は発行しない（TF不安定対策）
- ただしログは「start_pose基準（追従開始時のロボット姿勢基準）」の位置姿勢を保存する
  ※start_poseは数値的にのみ存在（originとして保持）

補足
- TF は map -> base_frame だけ参照（listen）する
- 経路可視化は /viz/path_markers, /viz/path_labels, /viz/active_path を使用（MarkerArray）
- 軌跡は /viz/robot_trajs に map 座標で描画し、CSVは start_pose基準で保存
"""

# =========================
# Standard Library Imports
# =========================
import math
import threading
import time
import datetime
import csv
import os
import copy

# =========================
# Third Party Imports
# =========================
import numpy as np
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import messagebox, ttk

# =========================
# ROS 2 Imports
# =========================
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data,
    ReliabilityPolicy,
)

# =========================
# ROS 2 Messages
# =========================
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
    Quaternion,
)
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import BatteryState, Imu
from nav2_msgs.action import FollowPath
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool

# =========================
# TF2 Imports
# =========================
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


# ==========================================
# Helper Functions
# ==========================================

def yaw_to_quat(z_yaw_rad: float) -> tuple:
    """Yaw角(rad)からクォータニオン(x, y, z, w)を生成する"""
    half = z_yaw_rad * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    return (0.0, 0.0, qz, qw)


def make_path(frame_id: str) -> tuple:
    """
    実験用の規定パスを生成する関数
    frame_id は「ローカル経路のラベル」に過ぎない（Nav2へはmapに変換して送る）
    Returns:
        (path_A, path_B, path_C): 生成された3種類のPathメッセージ
    """
    paths = []
    theta_list = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    l_segment = 3.0

    for theta in theta_list:
        # 1. 直進 0->3m
        x1 = np.linspace(0, 3, 300)
        y1 = np.zeros_like(x1)

        # 2. 斜め直線
        x2 = np.linspace(3.0, 3.0 + l_segment * math.cos(theta), 300)
        y2 = np.linspace(0.0, l_segment * math.sin(theta), 300)

        # 3. 終端直進
        x3 = np.linspace(
            3.0 + l_segment * math.cos(theta),
            6.0 + l_segment * math.cos(theta),
            300,
        )
        y3 = np.ones_like(x3) * l_segment * math.sin(theta)

        xs = np.concatenate([x1, x2, x3])
        ys = np.concatenate([y1, y2, y3])

        dx = np.gradient(xs)
        dy = np.gradient(ys)
        yaws = np.unwrap(np.arctan2(dy, dx))

        path = Path()
        path.header.frame_id = frame_id

        for x, y, yaw in zip(xs, ys, yaws):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            _, _, qz, qw = yaw_to_quat(yaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)

        paths.append(path)

    return (paths[0], paths[1], paths[2])


def make_iso_path(frame_id: str) -> tuple:
    """ISO用の規定パス（ローカル定義）"""
    paths = []
    Lu = 0.985

    # A: 直進
    x_A = np.linspace(0, 5 * Lu, 500)
    y_A = np.zeros_like(x_A)

    # B: 正方形
    x_B1 = np.linspace(0, 5 * Lu, 500)
    y_B1 = np.zeros_like(x_B1)
    y_B2 = np.linspace(0, -5 * Lu, 500)
    x_B2 = np.ones_like(y_B2) * 5 * Lu
    x_B3 = np.linspace(5 * Lu, 0, 500)
    y_B3 = np.ones_like(x_B3) * -5 * Lu
    y_B4 = np.linspace(-5 * Lu, 0, 500)
    x_B4 = np.zeros_like(y_B4)
    x_B = np.concatenate([x_B1, x_B2, x_B3, x_B4])
    y_B = np.concatenate([y_B1, y_B2, y_B3, y_B4])

    # C: 直進 + 円弧
    x_C1 = np.linspace(0, 5 * Lu, 500)
    y_C1 = np.zeros_like(x_C1)
    theta_list = np.linspace(0, np.pi / 2, 500)
    x_C2 = 5 * Lu * (1 + np.sin(theta_list))
    y_C2 = 5 * Lu * (np.cos(theta_list) - 1)
    x_C = np.concatenate([x_C1, x_C2])
    y_C = np.concatenate([y_C1, y_C2])

    x_list = [x_A, x_B, x_C]
    y_list = [y_A, y_B, y_C]

    for xs, ys in zip(x_list, y_list):
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        yaws = np.unwrap(np.arctan2(dy, dx))

        path = Path()
        path.header.frame_id = frame_id

        for x, y, yaw in zip(xs, ys, yaws):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            _, _, qz, qw = yaw_to_quat(yaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)

        paths.append(path)

    return (paths[0], paths[1], paths[2])


# ==========================================
# Main ROS 2 Node Class
# ==========================================

class FollowPathClient(Node):
    """
    - local_path を保持
    - send_path() で local_path -> map_path に変換し、Nav2へ送信
    - RVizへも map_path を描画（Nav2と一致）
    - ロボット軌跡は map で描画
    - CSV は start_pose基準（追従開始時のロボット姿勢基準）で保存
    """

    def __init__(self, local_path_frame_id: str = "local_path"):
        super().__init__("follow_path_gui_client")

        # --- Parameters ---
        self.local_path_frame_id = local_path_frame_id
        self.record_frequency = self.declare_parameter("record_frequency", 30).value
        self.data_dir = self.declare_parameter("data_dir", "/tmp").value
        self.map_frame_id = self.declare_parameter("map_frame_id", "map").value
        self.base_frame_id = self.declare_parameter("base_frame_id", "base_footprint").value
        self.experiment_name = self.declare_parameter("experiment_name", "dwpp").value  # dwpp or nelson

        # --- Internal State Variables ---
        self._reentrant_group = ReentrantCallbackGroup()
        self._current_goal_handle = None
        self._recording = False
        self._active_traj = None
        self.path_name = None
        self._traj_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._controller_id = None

        # start_origin (map基準) = 追従開始時のロボット姿勢
        self.start_origin_t_map = None  # np.array([x,y,z])
        self.start_origin_r_map = None  # scipy Rotation

        # 最新ロボット姿勢（map基準）
        self.current_pose_map = None  # geometry_msgs/Point相当(translation)
        self.current_quat_map = None  # geometry_msgs/Quaternion相当(rotation)

        # 最新ロボット姿勢（start_pose基準 = start_origin基準）
        self.current_pose_start = None  # geometry_msgs/Point
        self.current_quat_start = None  # geometry_msgs/Quaternion

        # ローカル経路（実験条件）
        # self._local_paths = make_path(self.local_path_frame_id)
        self._local_paths = make_iso_path(self.local_path_frame_id)

        # データ記録用バッファ
        self._reset_record_buffer()

        # 軌跡描画用点列（map）
        self._traj_points = {"PP": [], "APP": [], "RPP": [], "DWPP": []}
        self._traj_colors = {
            "PP": (0.0, 0.0, 1.0),
            "APP": (0.0, 0.5, 0.0),
            "RPP": (1.0, 0.647, 0.0),
            "DWPP": (1.0, 0.0, 0.0),
        }

        # 受信データキャッシュ
        self.current_odom = None
        self.current_cmd_vel_nav = None
        self.current_cmd_vel = None
        self.current_velocity_violation = False
        self.battery_voltage = float("nan")
        self.battery_current = float("nan")
        self.battery_percent = float("nan")
        self.imu_angular_vel_x = float("nan")
        self.imu_angular_vel_y = float("nan")
        self.imu_angular_vel_z = float("nan")
        self.imu_linear_acc_x = float("nan")
        self.imu_linear_acc_y = float("nan")
        self.imu_linear_acc_z = float("nan")

        # --- QoS Settings ---
        # RVizで「後からSubscribeしても見える」ようにするためのlatched QoS
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # --- TF Components (listen only) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Action Client ---
        self._client = ActionClient(self, FollowPath, "/follow_path")

        # --- Publishers ---
        self._initpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 10)
        self._label_pub = self.create_publisher(MarkerArray, "/viz/path_labels", latched_qos)
        self._path_markers_pub = self.create_publisher(MarkerArray, "/viz/path_markers", latched_qos)
        self._active_path_pub = self.create_publisher(MarkerArray, "/viz/active_path", latched_qos)
        self._traj_pub = self.create_publisher(MarkerArray, "/viz/robot_trajs", 10)

        # --- Subscribers ---
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self._cmd_vel_nav_sub = self.create_subscription(Twist, "/cmd_vel_nav", self._cmd_vel_nav_callback, 1)
        self._cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 1)
        self._velocity_violation_sub = self.create_subscription(
            Bool, "/constraints_violation_flag", self._velocity_violation_callback, 1
        )
        self.battery_state_sub = self.create_subscription(
            BatteryState, "/whill/states/battery_state", self._battery_state_callback, 1
        )
        self.imu_sub = self.create_subscription(Imu, "/ouster/imu", self._imu_callback, qos)

        # --- Background Threads & Timers ---
        # 起動直後の初期化
        threading.Thread(target=self._auto_publish_initial_pose, daemon=True).start()

        # パス（ローカル定義のサンプル）を map に載せ替えた形で常時表示するためのタイマ
        self._path_publish_ready_at = time.time() + 2.0
        self._path_publish_timer = self.create_timer(
            0.2, self._periodic_path_publish, callback_group=self._reentrant_group
        )

        # ロボット姿勢更新は必要なタイミングで取得する（常時タイマは使わない）
        self._get_robot_pose_timer = None

        # データ記録
        # self.record_timer = self.create_timer(
        #     1.0 / float(self.record_frequency),
        #     self._recording_loop,
        #     callback_group=self._reentrant_group,
        # )

        # 軌跡描画
        self._traj_draw_timer = self.create_timer(
            1.0 / 20.0,
            self._trajectory_draw_loop,
            callback_group=self._reentrant_group,
        )

    # =========================================================================
    # Subscriber Callbacks
    # =========================================================================

    def _imu_callback(self, msg: Imu):
        self.imu_angular_vel_x = msg.angular_velocity.x
        self.imu_angular_vel_y = msg.angular_velocity.y
        self.imu_angular_vel_z = msg.angular_velocity.z
        self.imu_linear_acc_x = msg.linear_acceleration.x
        self.imu_linear_acc_y = msg.linear_acceleration.y
        self.imu_linear_acc_z = msg.linear_acceleration.z

    def _battery_state_callback(self, msg: BatteryState):
        self.battery_voltage = msg.voltage
        self.battery_current = msg.current
        # percent は機体により入ってないことが多いので、そのまま保持
        if hasattr(msg, "percentage"):
            self.battery_percent = msg.percentage

    def _on_odom(self, msg: Odometry):
        self.current_odom = msg

    def _cmd_vel_nav_callback(self, msg: Twist):
        self.current_cmd_vel_nav = msg

    def _cmd_vel_callback(self, msg: Twist):
        self.current_cmd_vel = msg

    def _velocity_violation_callback(self, msg: Bool):
        self.current_velocity_violation = msg.data

    # =========================================================================
    # Pose Update (map listen + start conversion)
    # =========================================================================

    def _update_robot_pose_cache(self) -> bool:
        """ロボットの現在姿勢を map で取得し、start_pose基準（数値）も更新"""
        pose_map, quat_map = self._get_robot_pose(from_frame=self.map_frame_id)
        if pose_map is None:
            return False

        self.current_pose_map = pose_map
        self.current_quat_map = quat_map

        # start_origin が未設定なら start基準は計算できない
        if self.start_origin_t_map is None or self.start_origin_r_map is None:
            return True

        t_rel, r_rel = self._transform_map_pose_to_start(pose_map, quat_map)

        self.current_pose_start = Point(x=float(t_rel[0]), y=float(t_rel[1]), z=float(t_rel[2]))
        q = r_rel.as_quat()
        self.current_quat_start = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        return True

    def _get_robot_pose(self, from_frame: str):
        """指定フレームから見たロボット(base_frame_id)の位置姿勢を取得"""
        try:
            tf = self.tf_buffer.lookup_transform(from_frame, self.base_frame_id, rclpy.time.Time())
            return tf.transform.translation, tf.transform.rotation
        except TransformException as ex:
            # ログがうるさければdebugへ落としてもOK
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None, None

    # =========================================================================
    # Start Origin handling
    # =========================================================================

    def _set_start_origin_to_current_robot_pose(self) -> bool:
        """送信時点のロボット map 姿勢を start_origin として確定する"""
        pose_map, quat_map = self._get_robot_pose(from_frame=self.map_frame_id)
        if pose_map is None:
            return False
        self.start_origin_t_map = np.array([pose_map.x, pose_map.y, pose_map.z], dtype=float)
        self.start_origin_r_map = R.from_quat([quat_map.x, quat_map.y, quat_map.z, quat_map.w])
        return True

    def _transform_map_pose_to_start(self, pose_map, quat_map):
        """
        map 基準の pose/orientation を start_origin 基準へ変換
        start_origin は map基準で保持されているため、
          start = inv(R0) * (map - t0)
        """
        t0 = self.start_origin_t_map
        r0 = self.start_origin_r_map

        t = np.array([pose_map.x, pose_map.y, pose_map.z], dtype=float)
        r = R.from_quat([quat_map.x, quat_map.y, quat_map.z, quat_map.w])

        t_rel = r0.inv().apply(t - t0)
        r_rel = r0.inv() * r
        return t_rel, r_rel

    def _transform_local_path_to_map_path(self, local_path: Path) -> Path:
        """
        local_path（ロボット原点基準）を、start_origin（map基準）で map_path に変換
        - 位置: p_map = R0 * p_local + t0
        - 姿勢: q_map = R0 * q_local
        """
        if self.start_origin_t_map is None or self.start_origin_r_map is None:
            raise RuntimeError("start origin is not set")

        t0 = self.start_origin_t_map
        r0 = self.start_origin_r_map

        map_path = Path()
        map_path.header.frame_id = self.map_frame_id

        for ps in local_path.poses:
            p_local = np.array([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z], dtype=float)
            p_map = r0.apply(p_local) + t0
            ql = ps.pose.orientation
            r_local = R.from_quat([ql.x, ql.y, ql.z, ql.w])
            r_map = r0 * r_local
            q_map = r_map.as_quat()

            ps_map = PoseStamped()
            ps_map.header.frame_id = self.map_frame_id
            ps_map.pose.position.x = float(p_map[0])
            ps_map.pose.position.y = float(p_map[1])
            ps_map.pose.position.z = float(p_map[2])
            ps_map.pose.orientation.x = float(q_map[0])
            ps_map.pose.orientation.y = float(q_map[1])
            ps_map.pose.orientation.z = float(q_map[2])
            ps_map.pose.orientation.w = float(q_map[3])
            map_path.poses.append(ps_map)

        return map_path

    # =========================================================================
    # Data Recording & CSV Logic
    # =========================================================================

    def _reset_record_buffer(self):
        self.record_state_dict = {
            "sec": [],
            "nsec": [],
            # start_pose基準（数値変換）
            "x": [],
            "y": [],
            "yaw": [],
            # 実測
            "v_real": [],
            "w_real": [],
            # 指令 (Control Server)
            "v_cmd": [],
            "w_cmd": [],
            # 指令 (Nav2 final)
            "v_nav": [],
            "w_nav": [],
            # flags
            "velocity_violation": [],
            # battery
            "battery_v": [],
            "battery_i": [],
            "battery_percent": [],
            # IMU
            "imu_ax": [],
            "imu_ay": [],
            "imu_az": [],
            "imu_vx": [],
            "imu_vy": [],
            "imu_vz": [],
        }

    def _recording_loop(self):
        """
        _recording True の間、start_pose基準の状態を記録。
        False になったらCSV保存。
        """
        if self.current_cmd_vel_nav is None or self.current_cmd_vel is None or self.current_odom is None:
            return
        if not self._update_robot_pose_cache():
            return
        if self.current_pose_start is None or self.current_quat_start is None:
            return

        data_snapshot = None
        with self._record_lock:
            if self._recording:
                # yaw (start基準)
                r = R.from_quat(
                    [
                        self.current_quat_start.x,
                        self.current_quat_start.y,
                        self.current_quat_start.z,
                        self.current_quat_start.w,
                    ]
                )
                yaw = float(r.as_euler("xyz")[2])

                now_ns = int(self.get_clock().now().nanoseconds)
                sec = now_ns * 1e-9
                nsec = now_ns % int(1e9)

                d = self.record_state_dict
                d["sec"].append(sec)
                d["nsec"].append(nsec)
                d["x"].append(float(self.current_pose_start.x))
                d["y"].append(float(self.current_pose_start.y))
                d["yaw"].append(yaw)

                d["v_real"].append(float(self.current_odom.twist.twist.linear.x))
                d["w_real"].append(float(self.current_odom.twist.twist.angular.z))

                d["v_cmd"].append(float(self.current_cmd_vel_nav.linear.x))
                d["w_cmd"].append(float(self.current_cmd_vel_nav.angular.z))

                d["v_nav"].append(float(self.current_cmd_vel.linear.x))
                d["w_nav"].append(float(self.current_cmd_vel.angular.z))

                d["velocity_violation"].append(bool(self.current_velocity_violation))

                d["battery_v"].append(float(self.battery_voltage))
                d["battery_i"].append(float(self.battery_current))
                d["battery_percent"].append(float(self.battery_percent))

                d["imu_ax"].append(float(self.imu_linear_acc_x))
                d["imu_ay"].append(float(self.imu_linear_acc_y))
                d["imu_az"].append(float(self.imu_linear_acc_z))
                d["imu_vx"].append(float(self.imu_angular_vel_x))
                d["imu_vy"].append(float(self.imu_angular_vel_y))
                d["imu_vz"].append(float(self.imu_angular_vel_z))
            else:
                if len(self.record_state_dict["x"]) > 0:
                    data_snapshot = {k: v.copy() for k, v in self.record_state_dict.items()}
                    self._reset_record_buffer()

        if data_snapshot is not None:
            self._save_to_csv(data_snapshot, self.path_name, self._controller_id)

    def _save_to_csv(self, data_snapshot: dict, traj_name: str, controller_id: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.data_dir}/{self.experiment_name}"
        traj_label = traj_name or "None"
        controller_label = controller_id or "None"
        filename = f"{dir_name}/{traj_label}_{controller_label}_{timestamp}.csv"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(filename, "w", newline="") as csvfile:
            fieldnames = list(data_snapshot.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            data_len = len(data_snapshot["x"])
            for i in range(data_len):
                row = {field: data_snapshot[field][i] for field in fieldnames}
                writer.writerow(row)

        self.get_logger().info(f"Saved trajectory data to '{filename}'")

    # =========================================================================
    # Action Client Logic (FollowPath)
    # =========================================================================

    def send_path(self, local_path_msg: Path, path_name: str, controller_id: str, goal_checker_id: str):
        """
        local_path を受け取り、
        1) 送信時点のロボット map 姿勢を start_origin として確定
        2) local_path -> map_path に数値変換
        3) Nav2 へ送信
        4) RViz へも同一の map_path を描画（Nav2と一致）
        """
        self.path_name = path_name

        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server `/follow_path` not available.")
            return

        # 1) start_origin 確定
        if not self._set_start_origin_to_current_robot_pose():
            self.get_logger().error("Failed to set start origin (map->base TF unavailable).")
            return

        # 2) local -> map
        try:
            map_path = self._transform_local_path_to_map_path(local_path_msg)
        except Exception as ex:
            self.get_logger().error(f"Failed to transform local path to map: {ex}")
            return

        # 3) RVizへ「実際に送るmap_path」を描画
        self.publish_active_path(map_path)

        # 4) Nav2へ送信
        goal = FollowPath.Goal()
        goal.path = map_path
        goal.controller_id = controller_id
        goal.goal_checker_id = goal_checker_id

        with self._traj_lock:
            self._active_traj = controller_id
            self._controller_id = controller_id
            self._traj_points[controller_id] = []  # 軌跡リセット
            self._recording = True

        send_future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(lambda f: self._goal_response_cb(f, controller_id))

    def _goal_response_cb(self, future, controller_id):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected.")
            with self._traj_lock:
                self._recording = False
            return

        self.get_logger().info(f"Goal accepted. Controller: {controller_id}")
        self._current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        with self._traj_lock:
            self._recording = False

        try:
            result = future.result()
            self.get_logger().info(f"Result: status={result.status}")
        except Exception as e:
            self.get_logger().error(f"Result callback failed: {e}")

    def cancel_current_goal(self):
        if self._current_goal_handle is None:
            self.get_logger().info("No active goal to cancel.")
            return

        self.get_logger().info("Canceling goal...")
        self._current_goal_handle.cancel_goal_async()
        with self._traj_lock:
            self._recording = False

    def _feedback_cb(self, feedback_msg):
        # 必要ならログ出し
        pass

    # =========================================================================
    # Visualization
    # =========================================================================

    def publish_paths_and_labels(self, map_path_A: Path, map_path_B: Path, map_path_C: Path):
        """map座標で定義されたパス3本とラベルを表示（サンプル）"""
        markers = MarkerArray()
        path_configs = [
            (map_path_A, (1.0, 0.0, 0.0), "PathA"),
            (map_path_B, (0.0, 1.0, 0.0), "PathB"),
            (map_path_C, (0.0, 0.0, 1.0), "PathC"),
        ]
        now = self.get_clock().now().to_msg()

        # Lines
        for mid, (path, color, _) in enumerate(path_configs):
            if not path.poses:
                continue

            m = Marker()
            m.header.frame_id = self.map_frame_id
            # latched目的なので stamp=0 でもOK（元コード踏襲）
            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0
            m.ns = "path_visualization"
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.020
            m.color.r = color[0]
            m.color.g = color[1]
            m.color.b = color[2]
            m.color.a = 1.0

            for ps in path.poses:
                m.points.append(ps.pose.position)

            markers.markers.append(m)

        self._path_markers_pub.publish(markers)

        # Labels
        labels = MarkerArray()
        for mid, (path, _, name) in enumerate(path_configs, start=100):
            m = Marker()
            m.header.frame_id = self.map_frame_id
            m.header.stamp = now
            m.ns = "path_labels"
            m.id = mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            if path.poses:
                m.pose.position.x = float(path.poses[-1].pose.position.x + 0.1)
                m.pose.position.y = float(path.poses[-1].pose.position.y + 0.1)
                m.pose.position.z = 0.3
            m.scale.z = 0.50
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.text = name
            labels.markers.append(m)

        self._label_pub.publish(labels)

    def publish_active_path(self, map_path: Path):
        """Nav2へ送るのと同一の map_path をRVizへ描画（強調表示）"""
        ma = MarkerArray()

        m = Marker()
        m.header.frame_id = self.map_frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "active_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        # 黒色
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.5

        for ps in map_path.poses:
            m.points.append(ps.pose.position)

        ma.markers.append(m)
        self._active_path_pub.publish(ma)

    def _trajectory_draw_loop(self):
        """ロボット軌跡を map 座標系で描画"""
        with self._traj_lock:
            active_traj = self._active_traj

        if active_traj is None:
            return
        if not self._update_robot_pose_cache():
            return

        if not self._recording:
            return
        
        with self._traj_lock:
            self._draw_robot_trajectory_map(self.current_pose_map, active_traj)

    def _draw_robot_trajectory_map(self, current_pos_map, traj_name):
        if traj_name not in self._traj_points:
            return

        pts = self._traj_points[traj_name]
        if self.start_origin_t_map is None or self.start_origin_r_map is None:
            return

        t_rel, _ = self._transform_map_pose_to_start(current_pos_map, self.current_quat_map)
        current_point = Point(x=float(t_rel[0]), y=float(t_rel[1]), z=float(t_rel[2]))

        if not pts or self._distance_2d(pts[-1], current_point) > 0.02:
            pts.append(current_point)
            if len(pts) > 5000:
                self._traj_points[traj_name] = pts[-2000:]

        marr = MarkerArray()
        now = self.get_clock().now().to_msg()
        mid = 0

        for name, points in self._traj_points.items():
            if len(points) < 2:
                continue

            r, g, b = self._traj_colors.get(name, (0.5, 0.5, 0.5))

            m = Marker()
            m.header.frame_id = self.map_frame_id
            m.header.stamp = now
            m.ns = "robot_trajectory"
            m.id = mid
            mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.040
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.95
            m.points = [self._transform_start_point_to_map(p) for p in points]
            marr.markers.append(m)

        self._traj_pub.publish(marr)

    def clear_trajectory(self):
        with self._traj_lock:
            for k in self._traj_points:
                self._traj_points[k] = []

        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self._traj_pub.publish(ma)
        self.get_logger().info("Cleared all robot trajectories.")

    # =========================================================================
    # Utils
    # =========================================================================

    def _distance_2d(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _transform_start_point_to_map(self, point_start: Point) -> Point:
        """start_origin基準の点を map に変換（描画用）"""
        t0 = self.start_origin_t_map
        r0 = self.start_origin_r_map
        p_local = np.array([point_start.x, point_start.y, point_start.z], dtype=float)
        p_map = r0.apply(p_local) + t0
        return Point(x=float(p_map[0]), y=float(p_map[1]), z=float(p_map[2]))

    # =========================================================================
    # Periodic tasks
    # =========================================================================

    def _auto_publish_initial_pose(self):
        """起動直後に initialpose を投げる（RViz/AMCL系の初期化補助）"""
        time.sleep(1.0)
        for _ in range(3):
            self.publish_initial_pose(0.0, 0.0, 0.0)
            time.sleep(0.5)

    def _periodic_path_publish(self):
        """
        ローカル経路を「現在のロボット姿勢を原点」として map に載せ替えた形で表示する。
        ※送信時にも同様に載せ替えるので、普段の目視確認用
        """
        with self._traj_lock:
            # if not self._recording:
            #     return
            pass

        if time.time() < self._path_publish_ready_at:
            return
        
        t0 = self.start_origin_t_map
        r0 = self.start_origin_r_map

        try:
            local_A, local_B, local_C = self._local_paths
            map_A = self._transform_local_path_to_map_path_with_given_origin(local_A, t0, r0)
            map_B = self._transform_local_path_to_map_path_with_given_origin(local_B, t0, r0)
            map_C = self._transform_local_path_to_map_path_with_given_origin(local_C, t0, r0)
            self.publish_paths_and_labels(map_A, map_B, map_C)
        except Exception as exc:
            self.get_logger().warn(f"Path publish failed: {exc}")

    def _transform_local_path_to_map_path_with_given_origin(self, local_path: Path, t0: np.ndarray, r0: R) -> Path:
        """可視化用：任意の(t0,r0)で local_path を map_path に変換"""
        map_path = Path()
        map_path.header.frame_id = self.map_frame_id

        for ps in local_path.poses:
            p_local = np.array([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z], dtype=float)
            p_map = r0.apply(p_local) + t0
            ql = ps.pose.orientation
            r_local = R.from_quat([ql.x, ql.y, ql.z, ql.w])
            r_map = r0 * r_local
            q_map = r_map.as_quat()

            ps_map = PoseStamped()
            ps_map.header.frame_id = self.map_frame_id
            ps_map.pose.position.x = float(p_map[0])
            ps_map.pose.position.y = float(p_map[1])
            ps_map.pose.position.z = float(p_map[2])
            ps_map.pose.orientation.x = float(q_map[0])
            ps_map.pose.orientation.y = float(q_map[1])
            ps_map.pose.orientation.z = float(q_map[2])
            ps_map.pose.orientation.w = float(q_map[3])
            map_path.poses.append(ps_map)

        return map_path

    def publish_initial_pose(self, x: float, y: float, yaw_rad: float):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        _, _, qz, qw = yaw_to_quat(yaw_rad)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)
        msg.pose.covariance = [0.0] * 36
        self._initpose_pub.publish(msg)


# ==========================================
# GUI Class
# ==========================================

class AppGUI:
    """Tkinterベースの操作盤"""

    FONT_L = ("Arial", 20)
    FONT_L_BOLD = ("Arial", 20, "bold")

    def __init__(self, node: FollowPathClient, local_paths_dict: dict):
        self.node = node
        self.local_paths_dict = local_paths_dict

        self.root = tk.Tk()
        self.root.title("FollowPath GUI (Nav2) - local->map")
        self.root.geometry("900x520")
        self.root.option_add("*Font", self.FONT_L)

        self._create_widgets()

    def _create_widgets(self):
        # 1. Controller Selection
        frm_ctrl = tk.Frame(self.root)
        frm_ctrl.pack(pady=15)

        tk.Label(frm_ctrl, text="Controller:").pack(side=tk.LEFT, padx=5)

        self.controller_var = tk.StringVar(value="PP")
        controllers = ["PP", "APP", "RPP", "DWPP"]
        cb = ttk.Combobox(
            frm_ctrl,
            textvariable=self.controller_var,
            values=controllers,
            state="readonly",
            width=10,
            font=self.FONT_L_BOLD,
        )
        cb.pack(side=tk.LEFT, padx=5)

        # 2. Buttons
        frm_btns = tk.Frame(self.root)
        frm_btns.pack(pady=10)

        tk.Button(
            frm_btns,
            text="Clear Trajectory",
            font=self.FONT_L_BOLD,
            command=self._on_clear_traj,
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            frm_btns,
            text="Update Path Origin",
            font=self.FONT_L_BOLD,
            command=self._on_update_origin,
        ).pack(side=tk.LEFT, padx=10)

        # 3. Path Selection
        tk.Label(self.root, text="Select Path to Start (local path)").pack(pady=(20, 5))
        frm_paths = tk.Frame(self.root)
        frm_paths.pack()

        keys = list(self.local_paths_dict.keys())
        for i, name in enumerate(keys):
            tk.Button(
                frm_paths,
                text=name,
                width=24,
                font=self.FONT_L_BOLD,
                command=lambda n=name: self._on_send(n),
            ).grid(row=i // 2, column=i % 2, padx=10, pady=10)

        # 4. STOP
        tk.Button(
            self.root,
            text="STOP / CANCEL",
            font=self.FONT_L_BOLD,
            fg="red",
            command=self._on_cancel,
        ).pack(pady=20)

    def _on_clear_traj(self):
        self.node.clear_trajectory()

    def _on_send(self, path_name: str):
        controller_id = self.controller_var.get()
        goal_checker = "goal_checker"

        self.node.get_logger().info(f"UI: Send '{path_name}' with '{controller_id}'")

        try:
            local_path = self.local_paths_dict[path_name]
            self.node.send_path(local_path, path_name, controller_id, goal_checker)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_cancel(self):
        self.node.cancel_current_goal()
    
    def _on_update_origin(self):
        self.node._set_start_origin_to_current_robot_pose()

    def run(self):
        self.root.mainloop()


# ==========================================
# Main
# ==========================================

def main():
    rclpy.init()

    # ローカル経路生成（ロボット原点基準）
    local_frame = "local_path"
    # local_A, local_B, local_C = make_path(local_frame)
    local_A, local_B, local_C = make_iso_path(local_frame)

    local_paths_registry = {
        "Path A": local_A,
        "Path B": local_B,
        "Path C": local_C,
    }

    node = FollowPathClient(local_path_frame_id=local_frame)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        gui = AppGUI(node, local_paths_registry)
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
