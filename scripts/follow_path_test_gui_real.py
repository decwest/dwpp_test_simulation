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
import statistics
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
import yaml
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
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
    Quaternion,
)
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import ComputePathToPose, FollowPath
from sensor_msgs.msg import BatteryState, Imu, LaserScan
from visualization_msgs.msg import Marker, MarkerArray

# ControllerComputation is only present on the Decwest navigation2 fork
# (feature/revision_experiments_jazzy). Timing recording is disabled gracefully
# when the message is unavailable (e.g. plain apt nav2_msgs).
try:
    from nav2_msgs.msg import ControllerComputation
except ImportError:  # pragma: no cover
    ControllerComputation = None

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


def load_map_path_csv(csv_path: str, frame_id: str = "map") -> Path:
    """保存済みの map 座標経路 (x,y,yaw CSV) を読み込む。"""
    path = Path()
    path.header.frame_id = frame_id
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        required = {"x", "y", "yaw"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(
                f"fixed plan {csv_path} must contain columns x,y,yaw"
            )
        for row in reader:
            x, y, yaw = float(row["x"]), float(row["y"]), float(row["yaw"])
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise RuntimeError(f"fixed plan {csv_path} contains non-finite values")
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = x
            ps.pose.position.y = y
            _, _, ps.pose.orientation.z, ps.pose.orientation.w = yaw_to_quat(yaw)
            path.poses.append(ps)
    if len(path.poses) < 2:
        raise RuntimeError(f"fixed plan {csv_path} has fewer than 2 poses")
    return path


def save_map_path_csv(path: Path, csv_path: str):
    """NavFn の map 座標経路を x,y,yaw CSV として原子的に保存する。"""
    if len(path.poses) < 2:
        raise RuntimeError("NavFn returned fewer than 2 poses")
    parent = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(parent, exist_ok=True)
    temporary = f"{csv_path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x", "y", "yaw"])
            for ps in path.poses:
                writer.writerow([
                    float(ps.pose.position.x),
                    float(ps.pose.position.y),
                    quat_to_yaw(ps.pose.orientation),
                ])
        os.replace(temporary, csv_path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build_local_paths(path_set: str, frame_id: str) -> dict:
    """
    path_set パラメータからローカル経路レジストリ {label: Path} を構築する。
    ラベルは記録ディレクトリ名になるため空白を含めない
    (解析側は {data_dir}/{experiment}/{label}/{controller}/*.csv を glob する)。
    """
    if path_set == "polyline":
        a, b, c = make_path(frame_id)
        return {"PathA": a, "PathB": b, "PathC": c}
    if path_set == "iso":
        a, b, c = make_iso_path(frame_id)
        return {"ISO_Straight": a, "ISO_Square": b, "ISO_Arc": c}
    if path_set == "corridor":
        # Corridor は初回送信時に NavFn で生成し、map 座標の固定経路として保存する。
        return {"Corridor": Path()}
    raise ValueError(f"unknown path_set: {path_set!r} (expected polyline / iso / corridor)")


# ==========================================
# Controller-agnostic constraint math
# (faithful port of dynamic_window_pure_pursuit_functions.hpp on
#  Decwest/navigation2 feature/revision_experiments_jazzy)
# ==========================================

# プラグイン内蔵ロガーと同一の35列 + 追加列(scan_min_dist)
RECORDER_CSV_COLUMNS = [
    "sec", "nsec",
    "odom_base_x", "odom_base_y", "odom_base_yaw",
    "map_odom_x", "map_odom_y", "map_odom_yaw",
    "map_base_x", "map_base_y", "map_base_yaw",
    "v_real", "w_real", "v_now", "w_now", "v_cmd", "w_cmd", "v_nav", "w_nav",
    "velocity_violation",
    "battery_v", "battery_i", "battery_percent",
    "imu_ax", "imu_ay", "imu_az", "imu_vx", "imu_vy", "imu_vz",
    "curvature", "dw_v_max", "dw_v_min", "dw_w_max", "dw_w_min", "v_reg",
    "scan_min_dist",
]

# experiment_trial_lib.CONTROLLER_TIMING_CSV_FIELDS と同一
TIMING_CSV_FIELDS = ["t", "sequence", "controller_id", "compute_time_ms", "success"]


def _window_1d(last_vel, max_vel, min_vel, max_accel, max_decel, dt, eps):
    """C++ compute_window / evaluate系ラムダ共通の1次元窓計算(三分岐)"""
    if last_vel > eps:
        cand_max = last_vel + max_accel * dt
        cand_min = last_vel + max_decel * dt
    elif last_vel < -eps:
        cand_max = last_vel - max_decel * dt
        cand_min = last_vel - max_accel * dt
    else:
        cand_max = last_vel + max_accel * dt
        cand_min = last_vel - max_accel * dt
    return min(cand_max, max_vel), max(cand_min, min_vel)


def compute_dynamic_window(v_now, w_now, limits, dt):
    """computeDynamicWindow の移植 (Eps=1e-3)。regulation 適用前の窓を返す。
    Returns: (dw_v_max, dw_v_min, dw_w_max, dw_w_min)"""
    eps = 1e-3
    v_max, v_min = _window_1d(
        v_now, limits["max_linear_vel"], limits["min_linear_vel"],
        limits["max_linear_accel"], limits["max_linear_decel"], dt, eps)
    w_max, w_min = _window_1d(
        w_now, limits["max_angular_vel"], limits["min_angular_vel"],
        limits["max_angular_accel"], limits["max_angular_decel"], dt, eps)
    return v_max, v_min, w_max, w_min


def evaluate_velocity_constraints(v_cmd, w_cmd, v_now, w_now, limits, dt):
    """evaluateVelocityConstraints の移植 (Eps=1e-2: 分岐閾値・許容誤差とも)"""
    eps = 1e-2

    def violated(cur, last, max_vel, min_vel, max_accel, max_decel):
        cand_max, cand_min = _window_1d(last, max_vel, min_vel, max_accel, max_decel, dt, eps)
        return cur > cand_max + eps or cur < cand_min - eps

    return violated(
        v_cmd, v_now, limits["max_linear_vel"], limits["min_linear_vel"],
        limits["max_linear_accel"], limits["max_linear_decel"],
    ) or violated(
        w_cmd, w_now, limits["max_angular_vel"], limits["min_angular_vel"],
        limits["max_angular_accel"], limits["max_angular_decel"],
    )


def calc_actual_velocity(v_cmd, w_cmd, v_now, w_now, limits, dt):
    """recordData 内 calc_actual_velocity の移植 (Eps=1e-3)。指令を実現可能窓へクリップ。"""
    eps = 1e-3

    def clip(cur, last, max_vel, min_vel, max_accel, max_decel):
        cand_max, cand_min = _window_1d(last, max_vel, min_vel, max_accel, max_decel, dt, eps)
        if cur > cand_max + eps:
            return cand_max
        if cur < cand_min - eps:
            return cand_min
        return cur

    v_nav = clip(v_cmd, v_now, limits["max_linear_vel"], limits["min_linear_vel"],
                 limits["max_linear_accel"], limits["max_linear_decel"])
    w_nav = clip(w_cmd, w_now, limits["max_angular_vel"], limits["min_angular_vel"],
                 limits["max_angular_accel"], limits["max_angular_decel"])
    return v_nav, w_nav


def quat_to_yaw(q) -> float:
    """geometry_msgs/Quaternion -> yaw [rad]"""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


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
        self.odom_frame_id = self.declare_parameter("odom_frame_id", "odom").value
        self.experiment_name = self.declare_parameter(
            "experiment_name", "mppi_obstacle_experiment/scratch"
        ).value
        # 経路セット: polyline (45/90/135度折れ線) / iso /
        # corridor (NavFn 固定大域経路)
        self.path_set = self.declare_parameter("path_set", "polyline").value
        self.scan_topic = self.declare_parameter("scan_topic", "/merged_scan_filtered").value
        self.corridor_plan_file = self.declare_parameter(
            "corridor_plan_file",
            "/home/ubuntu/ros2_ws/src/ytlab2_whill_modules/"
            "worlds/corridor/map/fixed_plan.csv",
        ).value
        self.corridor_planner_id = self.declare_parameter(
            "corridor_planner_id", "GridBased"
        ).value
        self.corridor_start = (
            float(self.declare_parameter("corridor_start_x", 0.0).value),
            float(self.declare_parameter("corridor_start_y", 0.0).value),
            math.radians(float(self.declare_parameter("corridor_start_yaw_deg", 0.0).value)),
        )
        self.corridor_goal = (
            float(self.declare_parameter("corridor_goal_x", 10.0).value),
            float(self.declare_parameter("corridor_goal_y", -0.2).value),
            math.radians(float(self.declare_parameter("corridor_goal_yaw_deg", 0.0).value)),
        )

        # 制約値(コントローラ非依存の violation / dynamic window 計算に使用)。
        # controller_server 側のプラグイン設定と一致させること。
        # limits_params_file (通常は Nav2 の params_path) を渡すと、そのファイルの
        # コントローラブロックから自動同期される(二重管理による食い違いを防止)。
        self.control_frequency = self.declare_parameter("control_frequency", 30.0).value
        self.limits = {
            "max_linear_vel": self.declare_parameter("max_linear_vel", 0.50).value,
            "min_linear_vel": self.declare_parameter("min_linear_vel", 0.0).value,
            "max_angular_vel": self.declare_parameter("max_angular_vel", 1.0).value,
            "min_angular_vel": self.declare_parameter("min_angular_vel", -1.0).value,
            "max_linear_accel": self.declare_parameter("max_linear_accel", 0.50).value,
            "max_linear_decel": self.declare_parameter("max_linear_decel", -0.50).value,
            "max_angular_accel": self.declare_parameter("max_angular_accel", 1.0).value,
            "max_angular_decel": self.declare_parameter("max_angular_decel", -1.0).value,
        }
        self.limits_params_file = self.declare_parameter("limits_params_file", "").value
        if self.limits_params_file:
            self._load_limits_from_params_file(self.limits_params_file)

        # --- Internal State Variables ---
        self._reentrant_group = ReentrantCallbackGroup()
        self._current_goal_handle = None
        self._recording = False
        self._active_traj = None
        self.path_name = None
        self._traj_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._controller_id = None
        self._corridor_map_path = None
        self._corridor_plan_request = None

        # start_origin (map基準) = 追従開始時のロボット姿勢
        self.start_origin_t_map = None  # np.array([x,y,z])
        self.start_origin_r_map = None  # scipy Rotation

        # 最新ロボット姿勢（map基準）
        self.current_pose_map = None  # geometry_msgs/Point相当(translation)
        self.current_quat_map = None  # geometry_msgs/Quaternion相当(rotation)

        # 最新ロボット姿勢（start_pose基準 = start_origin基準）
        self.current_pose_start = None  # geometry_msgs/Point
        self.current_quat_start = None  # geometry_msgs/Quaternion

        # ローカル経路（path_set パラメータで選択; label -> Path）
        self.local_paths_dict = build_local_paths(self.path_set, self.local_path_frame_id)

        # データ記録用バッファ
        self._reset_record_buffer()
        # 前周期の指令 (v_now/w_now)。プラグインの reset() と同様に送信開始時に 0 リセット。
        self._prev_cmd = (0.0, 0.0)

        # 軌跡描画用点列（map）
        controllers = ["PP", "APP", "RPP", "DWPP", "MPPI"]
        self._traj_points = {c: [] for c in controllers}
        self._traj_colors = {
            "PP": (0.0, 0.0, 1.0),
            "APP": (0.0, 0.5, 0.0),
            "RPP": (1.0, 0.647, 0.0),
            "DWPP": (1.0, 0.0, 0.0),
            "MPPI": (0.5, 0.0, 0.5),
        }

        # 受信データキャッシュ
        self.current_odom = None
        self.current_cmd_vel_nav = None
        self.current_cmd_vel = None
        self.scan_min_dist = float("nan")
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

        # --- Action Clients ---
        self._client = ActionClient(self, FollowPath, "/follow_path")
        self._planner_client = ActionClient(
            self, ComputePathToPose, "/compute_path_to_pose"
        )

        if self.path_set == "corridor" and os.path.exists(self.corridor_plan_file):
            try:
                self._corridor_map_path = load_map_path_csv(
                    self.corridor_plan_file, self.map_frame_id
                )
                self.get_logger().info(
                    f"Loaded fixed NavFn corridor plan: {self.corridor_plan_file} "
                    f"({len(self._corridor_map_path.poses)} poses)"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to load fixed corridor plan {self.corridor_plan_file}: {exc}"
                )

        # --- Publishers ---
        self._initpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 10)
        self._label_pub = self.create_publisher(MarkerArray, "/viz/path_labels", latched_qos)
        self._path_markers_pub = self.create_publisher(MarkerArray, "/viz/path_markers", latched_qos)
        self._active_path_pub = self.create_publisher(MarkerArray, "/viz/active_path", latched_qos)
        self._traj_pub = self.create_publisher(MarkerArray, "/viz/robot_trajs", 10)

        # --- Subscribers ---
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        # コントローラ生出力 (controller_server -> velocity_smoother 前段)。
        # 1周期1メッセージ = レコーダの行トリガなので取りこぼさない深さにする。
        record_qos = QoSProfile(
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._cmd_vel_nav_sub = self.create_subscription(
            Twist, "/cmd_vel_nav", self._cmd_vel_nav_callback, record_qos
        )
        self._cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 1)
        self.battery_state_sub = self.create_subscription(
            BatteryState, "/whill/states/battery_state", self._battery_state_callback, 1
        )
        self.imu_sub = self.create_subscription(Imu, "/ouster/imu", self._imu_callback, qos)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data
        )
        # 計算時間 (Decwest navigation2 fork の controller_server が発行)
        if ControllerComputation is not None:
            self._timing_sub = self.create_subscription(
                ControllerComputation,
                "/controller_server/computation_time",
                self._computation_time_callback,
                record_qos,
            )
        else:
            self._timing_sub = None
            self.get_logger().warn(
                "nav2_msgs/ControllerComputation not available - timing CSV disabled "
                "(build nav2_msgs from Decwest fork feature/revision_experiments_jazzy)"
            )

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

        # データ記録は /cmd_vel_nav 受信駆動 (_cmd_vel_nav_callback) で行う:
        # controller_server はゴール実行中 1 制御周期に 1 メッセージ発行するため、
        # プラグイン内蔵ロガーの「computeVelocityCommands 毎に 1 行」と同じ意味論になる。

        # 軌跡描画
        self._traj_draw_timer = self.create_timer(
            1.0 / 20.0,
            self._trajectory_draw_loop,
            callback_group=self._reentrant_group,
        )

    def _load_limits_from_params_file(self, path: str):
        """
        Nav2 params ファイルのコントローラブロックから制約値と制御周波数を読み、
        レコーダの violation / dynamic window 計算をプラグイン設定に同期する。
        (制限値の揃った最初のコントローラブロックを採用。MPPI ブロック等はスキップ)
        """
        try:
            with open(path) as f:
                doc = yaml.safe_load(f)
            cs = doc["controller_server"]["ros__parameters"]
            freq = cs.get("controller_frequency")
            keys = list(self.limits.keys())
            for name in cs.get("controller_plugins", []):
                block = cs.get(name)
                if isinstance(block, dict) and all(k in block for k in keys):
                    self.limits = {k: float(block[k]) for k in keys}
                    if freq:
                        self.control_frequency = float(freq)
                    self.get_logger().info(
                        f"Recorder limits synced from {path} ({name} block): "
                        f"{self.limits}, control_frequency={self.control_frequency}"
                    )
                    return
            self.get_logger().warn(
                f"No controller block with limit keys in {path}; using declared parameters"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to load limits from {path}: {exc}; using declared parameters"
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
        self._record_cycle(msg)

    def _cmd_vel_callback(self, msg: Twist):
        self.current_cmd_vel = msg

    def _scan_callback(self, msg: LaserScan):
        best = float("inf")
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and r < best:
                best = r
        self.scan_min_dist = best if math.isfinite(best) else float("nan")

    def _computation_time_callback(self, msg):
        with self._record_lock:
            if not self._recording:
                return
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            compute_time_ms = msg.duration.sec * 1e3 + msg.duration.nanosec * 1e-6
            self._timing_rows.append(
                [t, int(msg.sequence), str(msg.controller_id),
                 compute_time_ms, bool(msg.success)]
            )

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
        # 行リスト形式 (列順 = RECORDER_CSV_COLUMNS)
        self._record_rows = []
        self._timing_rows = []

    def _lookup_xy_yaw(self, from_frame: str, to_frame: str):
        """TF lookup -> (x, y, yaw)。失敗時は NaN 3つ組(プラグインの挙動と同じ)。"""
        try:
            tf = self.tf_buffer.lookup_transform(from_frame, to_frame, rclpy.time.Time())
            t = tf.transform.translation
            return float(t.x), float(t.y), quat_to_yaw(tf.transform.rotation)
        except TransformException:
            nan = float("nan")
            return nan, nan, nan

    def _record_cycle(self, cmd: Twist):
        """
        /cmd_vel_nav 1メッセージ = 1制御周期として1行記録する。
        プラグイン内蔵ロガー(recordData)と同一の35列スキーマ+追加列。
        コントローラ固有量 (curvature, v_reg) は NaN、dw_* / violation / v_nav は
        制約値からの再計算(コントローラ非依存)。
        """
        with self._record_lock:
            if not self._recording:
                self._prev_cmd = (float(cmd.linear.x), float(cmd.angular.z))
                return

            v_cmd = float(cmd.linear.x)
            w_cmd = float(cmd.angular.z)
            v_now, w_now = self._prev_cmd
            dt = 1.0 / float(self.control_frequency)

            now_ns = int(self.get_clock().now().nanoseconds)
            sec = now_ns // 1_000_000_000  # int (旧実装の float 秒バグを修正)
            nsec = now_ns % 1_000_000_000

            odom_x, odom_y, odom_yaw = self._lookup_xy_yaw(self.odom_frame_id, self.base_frame_id)
            mo_x, mo_y, mo_yaw = self._lookup_xy_yaw(self.map_frame_id, self.odom_frame_id)
            mb_x, mb_y, mb_yaw = self._lookup_xy_yaw(self.map_frame_id, self.base_frame_id)

            if self.current_odom is not None:
                v_real = float(self.current_odom.twist.twist.linear.x)
                w_real = float(self.current_odom.twist.twist.angular.z)
            else:
                v_real = w_real = float("nan")

            violation = evaluate_velocity_constraints(v_cmd, w_cmd, v_now, w_now, self.limits, dt)
            v_nav, w_nav = calc_actual_velocity(v_cmd, w_cmd, v_now, w_now, self.limits, dt)
            # 注: regulation (v_reg) 適用前の窓。プラグインCSVの dw_* は適用後なので
            # regulation が発動する区間では一致しない(violation 判定は regulation 非依存)。
            dw_v_max, dw_v_min, dw_w_max, dw_w_min = compute_dynamic_window(
                v_now, w_now, self.limits, dt
            )

            nan = float("nan")
            self._record_rows.append([
                sec, nsec,
                odom_x, odom_y, odom_yaw,
                mo_x, mo_y, mo_yaw,
                mb_x, mb_y, mb_yaw,
                v_real, w_real, v_now, w_now, v_cmd, w_cmd, v_nav, w_nav,
                1 if violation else 0,
                float(self.battery_voltage), float(self.battery_current), float(self.battery_percent),
                float(self.imu_linear_acc_x), float(self.imu_linear_acc_y), float(self.imu_linear_acc_z),
                float(self.imu_angular_vel_x), float(self.imu_angular_vel_y), float(self.imu_angular_vel_z),
                nan,  # curvature (controller-specific)
                dw_v_max, dw_v_min, dw_w_max, dw_w_min,
                nan,  # v_reg (controller-specific)
                float(self.scan_min_dist),
            ])
            self._prev_cmd = (v_cmd, w_cmd)

    def _stop_recording_and_save(self):
        """記録停止 + バッファがあればCSV保存(ロック外でファイルI/O)"""
        with self._record_lock:
            self._recording = False
            rows = self._record_rows
            timing_rows = self._timing_rows
            if rows or timing_rows:
                self._reset_record_buffer()
            else:
                return
        if rows or timing_rows:
            self._save_to_csv(rows, timing_rows, self.path_name, self._controller_id)

    def _save_to_csv(self, rows: list, timing_rows: list, traj_name: str, controller_id: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_label = (traj_name or "None").replace(" ", "")
        controller_label = controller_id or "None"
        dir_name = os.path.join(self.data_dir, self.experiment_name, traj_label, controller_label)
        os.makedirs(dir_name, exist_ok=True)

        basename = f"{traj_label}_{controller_label}_{timestamp}"
        filename = os.path.join(dir_name, f"{basename}.csv")
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(RECORDER_CSV_COLUMNS)
            for row in rows:
                writer.writerow(
                    [f"{v:.6f}" if isinstance(v, float) else v for v in row]
                )
        self.get_logger().info(f"Saved {len(rows)} rows to '{filename}'")

        if timing_rows:
            timing_dir = os.path.join(dir_name, "timing")
            os.makedirs(timing_dir, exist_ok=True)
            timing_file = os.path.join(timing_dir, f"{basename}_timing.csv")
            with open(timing_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(TIMING_CSV_FIELDS)
                writer.writerows(timing_rows)
            self.get_logger().info(f"Saved {len(timing_rows)} timing rows to '{timing_file}'")

        self._warn_if_unhealthy(rows, timing_rows)

    def _warn_if_unhealthy(self, rows: list, timing_rows: list):
        """行間隔・計時行数のウォッチドッグ(保存後の健全性警告)"""
        expected_dt = 1.0 / float(self.control_frequency)
        if len(rows) >= 10:
            ts = [r[0] + r[1] * 1e-9 for r in rows]
            dts = [b - a for a, b in zip(ts, ts[1:])]
            med = statistics.median(dts)
            if abs(med - expected_dt) > 0.2 * expected_dt:
                self.get_logger().warn(
                    f"Recorder health: median inter-row dt {med * 1e3:.1f} ms deviates "
                    f">20% from expected {expected_dt * 1e3:.1f} ms"
                )
        if self._timing_sub is not None and rows and len(timing_rows) < 0.95 * len(rows):
            self.get_logger().warn(
                f"Recorder health: timing rows {len(timing_rows)} < 95% of data rows {len(rows)}"
            )

    # =========================================================================
    # Action Client Logic (FollowPath)
    # =========================================================================

    def _make_corridor_plan_goal(self):
        """現在の終端 base_footprint 姿勢への明示始点つき NavFn goal を作る。"""
        goal = ComputePathToPose.Goal()
        stamp = self.get_clock().now().to_msg()
        sx, sy, syaw = self.corridor_start
        gx, gy, gyaw = self.corridor_goal

        goal.start.header.frame_id = self.map_frame_id
        goal.start.header.stamp = stamp
        goal.start.pose.position.x = sx
        goal.start.pose.position.y = sy
        _, _, goal.start.pose.orientation.z, goal.start.pose.orientation.w = yaw_to_quat(syaw)
        goal.goal.header.frame_id = self.map_frame_id
        goal.goal.header.stamp = stamp
        goal.goal.pose.position.x = gx
        goal.goal.pose.position.y = gy
        _, _, goal.goal.pose.orientation.z, goal.goal.pose.orientation.w = yaw_to_quat(gyaw)
        goal.planner_id = self.corridor_planner_id
        goal.use_start = True
        return goal

    def _send_corridor_path(self, path_name: str, controller_id: str, goal_checker_id: str):
        """固定経路を再利用し、未作成なら初回だけ NavFn で計画する。"""
        if self._corridor_map_path is not None:
            self._send_map_path(
                self._corridor_map_path, path_name, controller_id, goal_checker_id
            )
            return

        if os.path.exists(self.corridor_plan_file):
            try:
                self._corridor_map_path = load_map_path_csv(
                    self.corridor_plan_file, self.map_frame_id
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to load fixed corridor plan {self.corridor_plan_file}: {exc}"
                )
                return
            self._send_map_path(
                self._corridor_map_path, path_name, controller_id, goal_checker_id
            )
            return

        if self._corridor_plan_request is not None:
            self.get_logger().warn("NavFn corridor planning is already in progress")
            return
        if not self._planner_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Action server `/compute_path_to_pose` not available. "
                "Wait for planner_server to become active and retry."
            )
            return

        self._corridor_plan_request = (path_name, controller_id, goal_checker_id)
        sx, sy, _ = self.corridor_start
        gx, gy, _ = self.corridor_goal
        self.get_logger().info(
            f"Planning fixed corridor path once with {self.corridor_planner_id}: "
            f"({sx:.3f}, {sy:.3f}) -> ({gx:.3f}, {gy:.3f})"
        )
        future = self._planner_client.send_goal_async(self._make_corridor_plan_goal())
        future.add_done_callback(self._corridor_plan_goal_response_cb)

    def _corridor_plan_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to send NavFn goal: {exc}")
            self._corridor_plan_request = None
            return
        if not goal_handle.accepted:
            self.get_logger().error("NavFn corridor planning goal was rejected")
            self._corridor_plan_request = None
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._corridor_plan_result_cb)

    def _corridor_plan_result_cb(self, future):
        request = self._corridor_plan_request
        self._corridor_plan_request = None
        try:
            wrapped = future.result()
            if (wrapped.status != GoalStatus.STATUS_SUCCEEDED
                    or len(wrapped.result.path.poses) < 2):
                self.get_logger().error(
                    f"NavFn corridor planning failed (status={wrapped.status})"
                )
                return
            save_map_path_csv(wrapped.result.path, self.corridor_plan_file)
            # 保存した表現を読み直すことで、全コントローラ・全試行に完全に同じ
            # map 座標 Path を渡す。
            self._corridor_map_path = load_map_path_csv(
                self.corridor_plan_file, self.map_frame_id
            )
            self.get_logger().info(
                f"Saved fixed NavFn corridor plan: {self.corridor_plan_file} "
                f"({len(self._corridor_map_path.poses)} poses)"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to finish NavFn corridor planning: {exc}")
            return

        if request is not None:
            self._send_map_path(self._corridor_map_path, *request)

    def send_path(self, local_path_msg: Path, path_name: str, controller_id: str, goal_checker_id: str):
        """
        local_path を受け取り、
        1) 送信時点のロボット map 姿勢を start_origin として確定
        2) local_path -> map_path に数値変換
        3) Nav2 へ送信
        4) RViz へも同一の map_path を描画（Nav2と一致）
        """
        if self.path_set == "corridor":
            self._send_corridor_path(path_name, controller_id, goal_checker_id)
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

        self._send_map_path(
            map_path, path_name, controller_id, goal_checker_id,
            refresh_start_origin=False,
        )

    def _send_map_path(self, map_path: Path, path_name: str,
                       controller_id: str, goal_checker_id: str,
                       refresh_start_origin: bool = True):
        """準備済みの map 座標 Path を可視化・記録して FollowPath へ送る。"""
        self.path_name = path_name
        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server `/follow_path` not available.")
            return
        if refresh_start_origin and not self._set_start_origin_to_current_robot_pose():
            self.get_logger().error("Failed to set start origin (map->base TF unavailable).")
            return

        # RVizへ「実際に送るmap_path」を描画
        self.publish_active_path(map_path)

        # Nav2へ送信
        goal = FollowPath.Goal()
        goal.path = map_path
        goal.controller_id = controller_id
        goal.goal_checker_id = goal_checker_id

        with self._traj_lock:
            self._active_traj = controller_id
            self._controller_id = controller_id
            self._traj_points[controller_id] = []  # 軌跡リセット
        with self._record_lock:
            self._reset_record_buffer()
            self._prev_cmd = (0.0, 0.0)  # プラグイン reset() と同様に前周期指令を 0 に
            self._recording = True

        send_future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(lambda f: self._goal_response_cb(f, controller_id))

    def _goal_response_cb(self, future, controller_id):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected.")
            self._stop_recording_and_save()
            return

        self.get_logger().info(f"Goal accepted. Controller: {controller_id}")
        self._current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._stop_recording_and_save()

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
        self._stop_recording_and_save()

    def _feedback_cb(self, feedback_msg):
        # 必要ならログ出し
        pass

    # =========================================================================
    # Visualization
    # =========================================================================

    PATH_VIZ_COLORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

    def publish_paths_and_labels(self, named_map_paths: dict):
        """map座標に変換済みの経路群 {name: Path} とラベルを表示"""
        markers = MarkerArray()
        path_configs = [
            (path, self.PATH_VIZ_COLORS[i % len(self.PATH_VIZ_COLORS)], name)
            for i, (name, path) in enumerate(named_map_paths.items())
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

        if self.path_set == "corridor":
            if self._corridor_map_path is not None:
                self.publish_paths_and_labels({"Corridor": self._corridor_map_path})
            return

        t0 = self.start_origin_t_map
        r0 = self.start_origin_r_map

        try:
            named_map_paths = {
                name: self._transform_local_path_to_map_path_with_given_origin(local, t0, r0)
                for name, local in self.local_paths_dict.items()
            }
            self.publish_paths_and_labels(named_map_paths)
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

        controllers = (
            ["RPP", "DWPP"] if self.node.path_set == "corridor"
            else ["PP", "APP", "RPP", "DWPP", "MPPI"]
        )
        self.controller_var = tk.StringVar(value=controllers[0])
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
        path_label = (
            "Select Path to Start (fixed NavFn path)"
            if self.node.path_set == "corridor"
            else "Select Path to Start (local path)"
        )
        tk.Label(self.root, text=path_label).pack(pady=(20, 5))
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

    # ローカル経路は path_set パラメータからノード内で構築される
    # (polyline: 45/90/135度折れ線 / iso / corridor: NavFn 固定大域経路)
    node = FollowPathClient(local_path_frame_id="local_path")

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        gui = AppGUI(node, node.local_paths_dict)
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
