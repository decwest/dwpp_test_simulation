#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    dwpp_test_dir = get_package_share_directory("dwpp_test_simulation")
    ytlab_hsr_dir = get_package_share_directory("ytlab2_hsr_modules")
    hsrb_rosnav_dir = get_package_share_directory("hsrb_rosnav_config")

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    map_yaml = LaunchConfiguration("map")
    odom_topic = LaunchConfiguration("odom_topic")
    autostart = LaunchConfiguration("autostart")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    map_frame_id = LaunchConfiguration("map_frame_id")
    base_frame_id = LaunchConfiguration("base_frame_id")
    goal_checker_id = LaunchConfiguration("goal_checker_id")

    default_params_file = os.path.join(dwpp_test_dir, "params", "hsrb_dwvp_test_params.yaml")
    default_map_yaml = os.path.join(ytlab_hsr_dir, "maps", "map1.yaml")
    default_rviz_config = os.path.join(dwpp_test_dir, "rviz", "dwpp_test.rviz")
    default_data_dir = os.path.join(dwpp_test_dir, "data")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock (false for real robot)",
    )
    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Nav2 parameters YAML for DWVP real-robot validation",
    )
    declare_map = DeclareLaunchArgument(
        "map",
        default_value=default_map_yaml,
        description="Map yaml path for localization",
    )
    declare_odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="omni_base_controller/wheel_odom",
        description="Odometry topic for Nav2 and velocity smoother",
    )
    declare_autostart = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically startup the nav2 stack",
    )
    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz_config,
        description="RViz config file",
    )
    declare_map_frame_id = DeclareLaunchArgument(
        "map_frame_id",
        default_value="map",
        description="Global frame ID",
    )
    declare_base_frame_id = DeclareLaunchArgument(
        "base_frame_id",
        default_value="base_footprint",
        description="Robot base frame ID (used by GUI TF lookup)",
    )
    declare_goal_checker_id = DeclareLaunchArgument(
        "goal_checker_id",
        default_value="general_goal_checker",
        description="Goal checker ID used in FollowPath goals",
    )

    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(hsrb_rosnav_dir, "launch", "navigation_launch.py")),
        launch_arguments={
            "namespace": "",
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "map": map_yaml,
            "params_file": params_file,
            "odom_topic": odom_topic,
        }.items(),
    )

    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    gui_node = Node(
        package="dwpp_test_simulation",
        executable="follow_path_test_gui_real.py",
        name="follow_path_gui_real",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "nav2_params_file": params_file,
                "map_frame_id": map_frame_id,
                "base_frame_id": base_frame_id,
                "goal_checker_id": goal_checker_id,
                "data_dir": default_data_dir,
                "experiment_name": "dwvp_real",
            }
        ],
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_params)
    ld.add_action(declare_map)
    ld.add_action(declare_odom_topic)
    ld.add_action(declare_autostart)
    ld.add_action(declare_use_rviz)
    ld.add_action(declare_rviz)
    ld.add_action(declare_map_frame_id)
    ld.add_action(declare_base_frame_id)
    ld.add_action(declare_goal_checker_id)

    ld.add_action(nav2_navigation)
    ld.add_action(rviz_node)
    ld.add_action(gui_node)

    return ld
