#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    # ====== パス ======
    nav2_dir = get_package_share_directory('nav2_bringup')
    dwpp_test_dir = get_package_share_directory('dwpp_test_simulation')
    hsrb_sim_dir = get_package_share_directory('hsrb_gazebo_bringup')

    # ====== 引数 ======
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config = LaunchConfiguration('rviz_config')
    # params_file = LaunchConfiguration('params_file')

    # ====== Declare Arguments ======
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use sim time'
    )
    
    # default_params_file = os.path.join(dwpp_test_dir,  'params', 'hsrb_test_params.yaml')
    # declare_params = DeclareLaunchArgument('params_file', default_value=default_params_file, description='HSRB Nav2 parameters YAML')
    
    default_rviz = os.path.join(dwpp_test_dir, 'rviz', 'dwpp_test.rviz')
    declare_rviz = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='RViz config file'
    )
    

    # Spawn HSR and gazebo
    hsrb_simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(hsrb_sim_dir, 'launch', 'gazebo_bringup.launch.py')),
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        arguments=[
            '--x','0','--y','0','--z','0',
            '--roll','0','--pitch','0','--yaw','0',
            '--frame-id','map','--child-frame-id','odom'
        ],
        remappings=[('/tf_static','/tf_static'),('/tf','/tf')],
    )
    
    # ====== ros_gz_bridge 起動 ======
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='set_pose_bridge',
        output='screen',
        arguments=[
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose"
        ]
    )
    
    params_file = os.path.join(get_package_share_directory('dwpp_test_simulation'),  'params', 'hsrb_test_params.yaml')
    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(dwpp_test_dir, 'launch', 'nav2_bringup.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'params_file': params_file}.items()
    )
    
    gui_node = Node(
            package='dwpp_test_simulation',
            executable='follow_path_test_gui.py',
            name='follow_path_gui',
            output='screen',
            emulate_tty=True,
            parameters=[{'robot_model_name': "hsrb", 'world_model_name': "default", "record_frequency": 30, "data_dir": "/home/dev/ros2_ws/src/third_party/dwpp_test_simulation/data"}],
            remappings=[('/cmd_vel', '/omni_base_controller/cmd_vel')],
        )

    # ====== LaunchDescription ======
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    # ld.add_action(declare_params)
    ld.add_action(declare_rviz)
    
    ld.add_action(nav2_navigation)
    ld.add_action(hsrb_simulator)
    ld.add_action(rviz)
    ld.add_action(static_map_to_odom)
    ld.add_action(bridge)
    ld.add_action(gui_node)

    return ld
