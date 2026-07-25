from tkinter import filedialog
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
import math
from matplotlib import pyplot as plt
import glob
import os

# 参照経路の情報
def step_curves() -> list:
    
    paths = []
    
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l = 3.0
    
    for theta in theta_list:
        # section 1
        x1 = np.linspace(0, 1, 100)
        y1 = np.zeros_like(x1)
        
        # section 2
        x2 = np.linspace(1.0, 1.0+l*math.cos(theta), 100)
        y2 = np.linspace(0.0, l*math.sin(theta), 100)
        
        # section 3
        x3 = np.linspace(1.0+l*math.cos(theta), 4.0+l*math.cos(theta), 100)
        y3 = np.ones_like(x3) * l * math.sin(theta)
        
        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])
        path = np.c_[x, y]
        paths.append(path)
    
    return paths

# 経路追従誤差の算出
def calc_rmse(robot_path: np.ndarray, path: np.ndarray) -> float:

    # ロボット軌跡の各点とpath上の各点間の距離をまとめて計算 (scipyのcdistを使用)
    # distance_matrix は shape=(ロボット軌跡の点数, パスの点数)
    distance_matrix = cdist(robot_path, path, metric='euclidean')

    # 行方向(min)をとることで、各ロボット点に対する最小距離を取り出す
    min_distances = np.min(distance_matrix, axis=1)

    # RMSEを算出
    rmse = np.sqrt(np.mean(min_distances**2))
    
    return rmse

# 違反率の算出
def calc_violation_rate(violation_flags) -> float:
    violation_count = np.sum(violation_flags)
    total_count = len(violation_flags)
    violation_rate = violation_count / total_count
    return violation_rate

# filepath = filedialog.askopenfilename(
#         initialdir="/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb",
#         title="Select a file",
#         filetypes=(("csv files", "*.csv"), ("All files", "*.*"))
#     )

# print("Selected file:", filepath)
# filepath="/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/DWPP_20251121_070133.csv"

# フォルダを与えたら、globで全部抽出するように。で、txtに統計情報を保存
# globで全部抽出する版も後で作る
filedir = filedialog.askdirectory(initialdir = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb")
data_paths = glob.glob(filedir + "/*.csv")

for filepath in data_paths:
    print("Processing file:", filepath)
    df = pd.read_csv(filepath)
    
    # extract path name
    path_name = filepath.split("/")[-2]
    method_name = filepath.split("/")[-1].split("_")[0]

    # データの読み込み
    t = df["t"].values - df["t"].values[0]  # 開始時間を0に合わせる
    x = df["x"].values
    y = df["y"].values
    robot_path = np.c_[x, y]
    v = df["v"].values
    w = df["w"].values
    cmd_v = df["v_cmd"].values
    cmd_w = df["w_cmd"].values
    v_nav = df["v_nav"].values
    w_nav = df["w_nav"].values
    velocity_violation_flag = df["velocity_violation"].values

    pathA, pathB, pathC = step_curves()
    path_dict = {"PathA": pathA, "PathB": pathB, "PathC": pathC}
    
    rmse = calc_rmse(robot_path, path_dict[path_name])
    violation_rate = calc_violation_rate(velocity_violation_flag)
    print(f"経路追従誤差RMSE ({path_name}): {rmse:.4f} m")
    print(f"違反率: {violation_rate:.4f}")
    
    # txtに書き出し
    txt_filename = os.path.dirname(filepath) + "/result.txt"
    with open(txt_filename, "a") as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"RMSE: {rmse:.4f} m\n")
        f.write(f"Violation Rate: {violation_rate:.4f}\n")
        f.write("\n")

    # 速度プロファイルの図示
    plt.figure(figsize=(10, 6))
    plt.plot(t, v, label="actual", color='blue')
    plt.plot(t, cmd_v, label="reference", color='red')
    plt.xlabel("Time [s]")
    plt.ylabel("Linear Velocity [m/s]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.path.dirname(filepath)}/{method_name}_velocity_profile.png")

    plt.figure(figsize=(10, 6))
    plt.plot(t, w, label="actual", color='blue')
    plt.plot(t, cmd_w, label="reference", color='red')
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{os.path.dirname(filepath)}/{method_name}_angular_velocity_profile.png")
