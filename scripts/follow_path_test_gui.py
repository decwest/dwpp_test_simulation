#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from scipy.spatial.transform import Rotation as R
import datetime
import csv

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path, Odometry
from nav2_msgs.action import FollowPath
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool

from ros_gz_interfaces.srv import SetEntityPose  # Gazebo Sim (Ignition) の set_pose サービス

import os
from ament_index_python.packages import get_package_share_directory


def yaw_to_quat(z_yaw_rad: float):
    half = z_yaw_rad * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    return (0.0, 0.0, qz, qw)


def make_path(frame_id: str):
    paths = []
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l_segment = 3.0

    for theta in theta_list:
        x1 = np.linspace(0, 3, 300)
        y1 = np.zeros_like(x1)
        
        # 2. 斜め直線 (角度thetaで長さl)
        # Note: 実際には直線補間だが、ここでは簡易的に生成
        x2 = np.linspace(3.0, 3.0 + l_segment * math.cos(theta), 300)
        y2 = np.linspace(0.0, l_segment * math.sin(theta), 300)
        
        # 3. 終端直進 (さらに3m進む)
        x3 = np.linspace(
            3.0 + l_segment * math.cos(theta), 
            6.0 + l_segment * math.cos(theta), 300)
        y3 = np.ones_like(x3) * l_segment * math.sin(theta)

        # 結合
        xs = np.concatenate([x1, x2, x3])
        ys = np.concatenate([y1, y2, y3])
        dx = np.gradient(xs); dy = np.gradient(ys)
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

    path_A, path_B, path_C = paths
    return (path_A, path_B, path_C)


class FollowPathClient(Node):
    def __init__(self, frame_id: str = "map"):
        super().__init__('follow_path_gui_client')
        self._client = ActionClient(self, FollowPath, '/follow_path')
        self._current_goal_handle = None
        self._frame_id = frame_id
        
        # parameters
        self.robot_model_name = self.declare_parameter('robot_model_name', "turtlebot3_waffle").value
        self.world_model_name = self.declare_parameter('world_model_name', "empty").value
        self.record_frequency = self.declare_parameter('record_frequency', 30).value
        self.data_dir = self.declare_parameter('data_dir', '/tmp').value

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
        self._traj_points = {
            'PP':   [],
            'APP':  [],
            'RPP':  [],
            'DWPP': []
        }
        self._active_traj = None          # 現在アクティブな手法名（send_path時に設定）
        self._traj_frame_id = 'odom'
        self._traj_lock = threading.Lock()

        # 手法→色マップ（R,G,B）
        self._traj_colors = {
            'PP':   (0.0, 0.0, 1.0),   # blue
            'APP':  (0.0, 1.0, 0.0),   # green
            'RPP':  (1.0, 1.0, 0.0),   # yellow
            'DWPP': (1.0, 0.0, 0.0)    # red
        }

        # 走行中のみ記録するためのフラグ
        self._recording = False

        # /odom 購読（SensorData QoS）
        self.current_odom = None
        self._odom_sub = self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        # control serverが出力する速度指令値の保存用
        self.current_cmd_vel_nav = None
        self._cmd_vel_nav_sub = self.create_subscription(Twist, '/cmd_vel_nav', self._cmd_vel_nav_callback, 1)
        # Nav2が出力する速度指令地の保存用
        self.current_cmd_vel = None
        self._cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, 1)
        # 速度違反フラグの保存用
        self.current_velocity_violation = False
        self._velocity_violation_sub = self.create_subscription(Bool, '/constraints_violation_flag', self._velocity_violation_callback, 1)

        # Gazebo warp clients
        self._gz_setpose_cli = None  # /world/<world>/set_pose 用

        # 起動直後：初期姿勢 & tb3 ワープ & パス定期描画
        threading.Thread(target=self._auto_publish_initial_pose, daemon=True).start()
        threading.Thread(target=self._auto_warp, daemon=True).start()
        threading.Thread(target=self._periodic_path_publish, daemon=True).start()
        # 記録用のタイマー割込み
        self.record_rate = self.create_rate(self.record_frequency)  # 30 Hz
        threading.Thread(target=self._recording_timer_callback, daemon=True).start()

    def _velocity_violation_callback(self, msg: Bool):
        self.current_velocity_violation = msg.data

    def _cmd_vel_nav_callback(self, msg: Twist):
        self.current_cmd_vel_nav = msg

    def _cmd_vel_callback(self, msg: Twist):
        self.current_cmd_vel = msg

    def _recording_timer_callback(self):
        self.record_state_dict = {"t": [], "x": [], "y": [], "yaw": [], "v": [], "w": [], "v_cmd": [], "w_cmd": [], "v_nav": [], "w_nav": [], "velocity_violation": []}
        while rclpy.ok():
            # aaa
            if  self.current_cmd_vel_nav is None or self.current_cmd_vel is None or self.current_odom is None:
                continue
            
            if self._recording:
                # self.get_logger().info(f"Now recording {self._active_traj} trajectory...")
                
                # actual position
                x = self.current_odom.pose.pose.position.x
                y = self.current_odom.pose.pose.position.y
                
                qx = self.current_odom.pose.pose.orientation.x
                qy = self.current_odom.pose.pose.orientation.y
                qz = self.current_odom.pose.pose.orientation.z
                qw = self.current_odom.pose.pose.orientation.w
                r = R.from_quat([qx, qy, qz, qw])
                yaw = r.as_euler('xyz')[2]
                
                # actual velocity
                v = self.current_odom.twist.twist.linear.x
                w = self.current_odom.twist.twist.angular.z
                
                # commanded velocity from control server
                v_cmd = self.current_cmd_vel_nav.linear.x
                w_cmd = self.current_cmd_vel_nav.angular.z
                
                # commanded velocity from Nav2
                v_nav = self.current_cmd_vel.linear.x
                w_nav = self.current_cmd_vel.angular.z
                
                velocity_violation = self.current_velocity_violation
                
                # print(self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9)
                self.record_state_dict["t"].append(self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9)
                self.record_state_dict["x"].append(x)
                self.record_state_dict["y"].append(y)
                self.record_state_dict["yaw"].append(yaw)
                self.record_state_dict["v"].append(v)
                self.record_state_dict["w"].append(w)
                self.record_state_dict["v_cmd"].append(v_cmd)
                self.record_state_dict["w_cmd"].append(w_cmd)
                self.record_state_dict["v_nav"].append(v_nav)
                self.record_state_dict["w_nav"].append(w_nav)
                self.record_state_dict["velocity_violation"].append(velocity_violation)
                
            else:
                # 保存データがあるなら
                if len(self.record_state_dict["x"]) > 0:
                    # save to csv file
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dir_name = f"{self.data_dir}/{self.robot_model_name}"
                    filename = f"{dir_name}/{self._active_traj}_{timestamp}.csv"
                    if os.path.exists(dir_name) == False:
                        os.makedirs(dir_name, exist_ok=True)
                    with open(filename, 'w', newline='') as csvfile:
                        fieldnames = ["t", "x", "y", "yaw", "v", "w", "v_cmd", "w_cmd", "v_nav", "w_nav", "velocity_violation"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for i in range(len(self.record_state_dict["x"])):
                            writer.writerow({field: self.record_state_dict[field][i] for field in fieldnames})
                    self.get_logger().info(f"Saved trajectory data to '{filename}'")
                    
                    # clear the record
                    self.record_state_dict = {"t": [], "x": [], "y": [], "yaw": [], "v": [], "w": [], "v_cmd": [], "w_cmd": [], "v_nav": [], "w_nav": [], "velocity_violation": []}
                
            self.record_rate.sleep()

    # ===== RViz 可視化：パス & ラベル =====
    def publish_paths_and_labels(self, path_A: Path, path_B: Path, path_C: Path):
        # Path visualization markers with different colors
        markers = MarkerArray()
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # Red, Green, Blue
        paths = [('Path A', path_A), ('Path B', path_B), ('Path C', path_C)]
        now = self.get_clock().now().to_msg()

        for mid, ((name, path), color) in enumerate(zip(paths, colors)):
            if not path.poses:
                continue

            marker = Marker()
            marker.header.frame_id = path.header.frame_id
            marker.header.stamp = now
            marker.ns = 'path_visualization'
            marker.id = mid
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # Line width
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            for pose_stamped in path.poses:
                marker.points.append(pose_stamped.pose.position)

            markers.markers.append(marker)

        self._path_markers_pub.publish(markers)

        # Labels (TEXT_VIEW_FACING)
        labels = MarkerArray()
        for mid, (name, p) in enumerate([('PathA', path_A), ('PathB', path_B), ('PathC', path_C)], start=1):
            m = Marker()
            m.header.frame_id = p.header.frame_id
            m.header.stamp = now
            m.ns = 'path_labels'
            m.id = mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            if p.poses:
                m.pose.position.x = p.poses[-1].pose.position.x + 0.1
                m.pose.position.y = p.poses[-1].pose.position.y + 0.1
            m.pose.position.z = 0.3
            m.scale.z = 0.25
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0
            m.text = name
            labels.markers.append(m)
        self._label_pub.publish(labels)

    # ===== Robot trajectory =====
    def _on_odom(self, msg: Odometry):
        self.current_odom = msg
        with self._traj_lock:
            # 走行中でなければ記録しない（Warp などはここで無視）
            if not self._recording:
                return

            if self._traj_frame_id != msg.header.frame_id:
                # フレーム変更に追従（全軌跡クリア）
                for k in self._traj_points:
                    self._traj_points[k] = []
                self._traj_frame_id = msg.header.frame_id

            # 追従手法が未選択なら何もしない
            if self._active_traj not in self._traj_points:
                return

            current_pos = msg.pose.pose.position
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
                m.scale.x = 0.035
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95
                m.points = points.copy()
                marr.markers.append(m)

            self._traj_pub.publish(marr)

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

        self.get_logger().info("Cleared ALL robot trajectories (PP/APP/RPP/DWPP).")

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
        path_A, path_B, path_C = make_path(self._frame_id)
        while rclpy.ok():
            try:
                self.publish_paths_and_labels(path_A, path_B, path_C)
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().warn(f"Periodic path publish failed: {e}")
                time.sleep(5.0)

    # ===== follow_path action =====
    def send_path(self, path_msg: Path, controller_id: str, goal_checker_id: str):
        if not self._client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("`/follow_path` action server not available.")

        goal = FollowPath.Goal()
        goal.path = path_msg
        goal.controller_id = controller_id
        goal.goal_checker_id = goal_checker_id

        with self._traj_lock:
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
        if gh is None:
            self.get_logger().info('No active goal to cancel.')
            return
        cancel_future = gh.cancel_goal_async()
        cancel_future.add_done_callback(lambda _: self.get_logger().info('Cancel request sent.'))
        # キャンセルしたら記録OFF
        with self._traj_lock:
            self._recording = False

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
        except Exception as e:
            self.get_logger().warn(f'Result callback error: {e}')


class AppGUI:
    def __init__(self, node: FollowPathClient, paths_dict: dict, frame_id: str):
        self.node = node
        self.paths_dict = paths_dict
        self.frame_id = frame_id

        self.root = tk.Tk()
        self.root.title("FollowPath GUI (Nav2)")
        self.root.geometry("800x400")
        
        self.robot_model_name = self.node.robot_model_name
        self.world_model_name = self.node.world_model_name
        
        # Set larger default font for the whole application
        default_font = ("Arial", 20)
        self.root.option_add("*Font", default_font)

        tk.Label(self.root, text=f"frame_id: {self.frame_id}", font=("Arial", 20)).pack(pady=8)

        frm = tk.Frame(self.root); frm.pack(pady=8)

        tk.Label(frm, text="Controller:", font=("Arial", 20)).grid(row=0, column=0, sticky="e")
        self.controller_var = tk.StringVar(value="PP")
        self.controller_cb = ttk.Combobox(frm, textvariable=self.controller_var,
                                          values=["PP", "APP", "RPP", "DWPP"], state="readonly", width=10, font=("Arial", 20, "bold"))
        self.controller_cb.grid(row=0, column=1, padx=6)

        self.goal_checker_id = "general_goal_checker"

        # Buttons row
        btn_row = tk.Frame(self.root); btn_row.pack(pady=(8, 12))
        tk.Button(btn_row, text="Warp Robot (0,0,0)", font=("Arial", 20, "bold"), command=self._on_warp).grid(row=0, column=0, padx=8)
        tk.Button(btn_row, text="Clear Trajectory", font=("Arial", 20, "bold"), command=self._on_clear_traj).grid(row=0, column=1, padx=8)

        tk.Label(self.root, text="Paths", font=("Arial", 20)).pack(pady=(10, 4))
        btns = tk.Frame(self.root); btns.pack()

        for i, name in enumerate(self.paths_dict.keys()):
            tk.Button(btns, text=name, width=20, font=("Arial", 20, "bold"),
                      command=lambda n=name: self._on_send(n)).grid(row=i // 2, column=i % 2, padx=8, pady=8)

        tk.Button(self.root, text="Cancel", font=("Arial", 20, "bold"), command=self._on_cancel).pack(pady=(12, 8))

        # 初回：PathとラベルをPublish
        self.node.publish_paths_and_labels(
            self.paths_dict["Path A (45 deg)"],
            self.paths_dict["Path B (90 deg)"],
            self.paths_dict["Path C (135 deg)"],
        )

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
            path_msg = self.paths_dict[path_name]
            self.node.get_logger().info(f"Sending path '{path_name}' using controller '{controller_id}'")
            self.node.send_path(path_msg, controller_id, self.goal_checker_id)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_cancel(self):
        self.node.cancel_current_goal()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()

    frame_id = "map"
    world_name = "empty"
    path_A, path_B, path_C = make_path(frame_id)

    paths = {
        "Path A (45 deg)": path_A,
        "Path B (90 deg)": path_B,
        "Path C (135 deg)": path_C,
    }

    node = FollowPathClient(frame_id=frame_id)
    # === 安全な executor / spin ===
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        gui = AppGUI(node, paths, frame_id)
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
