import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

# Set font to Times New Roman and font size to 22
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22

# Path C PP
# PathC_PP_path = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/PathC/PP_20251121_071847.csv"
PathC_PP_path = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/PathC/DWPP_20251121_072134.csv"
PathC_PP_data = pd.read_csv(PathC_PP_path)

t = PathC_PP_data["t"].values - PathC_PP_data["t"].values[0]
v = PathC_PP_data["v"].values
w = PathC_PP_data["w"].values
v_cmd = PathC_PP_data["v_cmd"].values
w_cmd = PathC_PP_data["w_cmd"].values


# Create subplots for translational and rotational velocities (horizontal layout)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# Plot translational velocity
ax1.plot(t, v_cmd, '-', color='red', label='Reference', linewidth=3, alpha=0.8)
# ax1.plot(timestamps, cmd_linear, '-', color='blue', label='Command', linewidth=3)
ax1.plot(t, v, '-', color='blue', label='Actual', linewidth=3)
# Add velocity limit line
ax1.axhline(y=0.22, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Max Velocity')

# Plot rotational velocity
ax2.plot(t, w_cmd, '-', color='red', label='Reference', linewidth=3, alpha=0.8)
# ax2.plot(timestamps, cmd_angular, '-', color='blue', label='Command', linewidth=3)
ax2.plot(t, w, '-', color='blue', label='Actual', linewidth=3)
# Add velocity limit lines
ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Max Velocity')
ax2.axhline(y=-0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Min Velocity')

# Set labels and legends
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('Linear Velocity [m/s]')
ax1.set_yticks([0, 0.10, 0.20])
ax1.grid(True, alpha=0.3)
# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# ax2.set_xlabel('Time [s]')
# ax2.set_ylabel('Angular Velocity [rad/s]')
ax2.set_ylim(-0.7, 0.7)
ax2.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax2.grid(True, alpha=0.3)
# ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'velocity_profile_pp.png', dpi=300, bbox_inches='tight')
plt.show()

# 経路の描画
# Path Cのデータ
PathC_PP_path = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/PathC/PP_20251121_071847.csv"
PathC_PP_data = pd.read_csv(PathC_PP_path)
PathC_APP_path = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/PathC/APP_20251121_071940.csv"
PathC_APP_data = pd.read_csv(PathC_APP_path)
PathC_RPP_path = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/PathC/RPP_20251121_072039.csv"
PathC_RPP_data = pd.read_csv(PathC_RPP_path)
PathC_DWPP_path = "/home/decwest/decwest_workspace/ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation/data/hsrb/PathC/DWPP_20251121_072134.csv"
PathC_DWPP_data = pd.read_csv(PathC_DWPP_path)

pp_x = PathC_PP_data["x"].values
pp_y = PathC_PP_data["y"].values
app_x = PathC_APP_data["x"].values
app_y = PathC_APP_data["y"].values
rpp_x = PathC_RPP_data["x"].values
rpp_y = PathC_RPP_data["y"].values
dwpp_x = PathC_DWPP_data["x"].values
dwpp_y = PathC_DWPP_data["y"].values

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

PathA, PathB, PathC = step_curves()

fig = plt.figure(figsize=(6.5,6.5))
ax = fig.add_subplot(1, 1, 1)

# Plot robot trajectory (rotate: X->Y, Y->-X for X-up, Y-left coordinate system)
ax.plot(pp_y, [-x for x in pp_x], color="blue", label='PP', linewidth=4.5)
ax.plot(app_y, [-x for x in app_x], color="green", label='APP', linewidth=4.5)
ax.plot(rpp_y, [-x for x in rpp_x], color="orange", label='RPP', linewidth=4.5)
ax.plot(dwpp_y, [-x for x in dwpp_x], color="red", label='DWPP', linewidth=4.5)

ref_x = [point[0] for point in PathC]
ref_y = [point[1] for point in PathC]
ax.plot(ref_y, [-x for x in ref_x], 'k--', label='Reference Path', linewidth=3, alpha=0.7)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.invert_xaxis()
ax.invert_yaxis()
plt.tight_layout()

plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
plt.show()