#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from nav2_common.launch import LaunchConfigAsBool
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    # ====== パス ======
    nav2_dir = get_package_share_directory('nav2_bringup')
    sim_dir = get_package_share_directory('nav2_minimal_tb3_sim')
    dwpp_test_dir = get_package_share_directory('dwpp_test_simulation')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')

    # ====== 引数 ======
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfigAsBool('use_sim_time')  # ここでは spawn にだけ渡す
    world = LaunchConfiguration('world')               # 既定: 床だけ
    robot_name = LaunchConfiguration('robot_name')
    robot_sdf = LaunchConfiguration('robot_sdf')
    bridge_yaml = LaunchConfiguration('bridge_yaml')
    rviz_config = LaunchConfiguration('rviz_config')
    params_file = LaunchConfiguration('params_file')
    default_urdf_xacro  = os.path.join(sim_dir, 'urdf', 'turtlebot3_waffle.urdf')
    urdf_xacro    = LaunchConfiguration('urdf_xacro')
    

    # ====== Declare Arguments ======
    declare_namespace = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use sim time'
    )
    
    default_params_file = os.path.join(dwpp_test_dir,  'params', 'test_params.yaml')
    declare_params = DeclareLaunchArgument('params_file', default_value=default_params_file, description='Nav2 parameters YAML')
    
    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(dwpp_test_dir, 'world', 'empty.sdf'),
        description='World file (SDF)',
    )
    declare_robot_name = DeclareLaunchArgument(
        'robot_name', default_value='turtlebot3_waffle', description='Robot name'
    )
    declare_robot_sdf = DeclareLaunchArgument(
        'robot_sdf',
        default_value=os.path.join(dwpp_test_dir, 'urdf', 'gz_waffle.sdf.xacro'),
        description='Robot SDF(Xacro) path for Gazebo spawn',
    )
    
    default_bridge_yaml = os.path.join(dwpp_test_dir, 'config', 'ros_gz_bridge.yaml')
    declare_yaml = DeclareLaunchArgument(
        'bridge_yaml', default_value=default_bridge_yaml,
        description='ros_gz_bridge YAML config'
    )
    
    default_rviz = os.path.join(dwpp_test_dir, 'rviz', 'dwpp_test.rviz')
    declare_rviz = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='RViz config file'
    )
    declare_urdfx  = DeclareLaunchArgument('urdf_xacro',  default_value=default_urdf_xacro,  description='TB3 xacro for RSP')
    

    # ====== Gazebo GUI起動（GUIのみ） ======
    # gz_sim.launch.py の引数は文字列で渡すのが確実: "-r -v 4 <world>"
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': PythonExpression(["'-r -v 4 ' + '", world, "'"]),
            'gui': 'true',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # ====== TurtleBot3 スポーン ======
    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sim_dir, 'launch', 'spawn_tb3.launch.py')),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,  # nav2_minimal_tb3_sim 側の引数型に合わせて渡す
            'robot_name': robot_name,
            'robot_sdf': robot_sdf,
            'x_pose': LaunchConfiguration('x_pose', default='0.0'),
            'y_pose': LaunchConfiguration('y_pose', default='0.0'),
            'z_pose': LaunchConfiguration('z_pose', default='1.0'),
            'roll':   LaunchConfiguration('roll', default='0.0'),
            'pitch':  LaunchConfiguration('pitch', default='0.0'),
            'yaw':    LaunchConfiguration('yaw', default='0.0'),
        }.items(),
    )
    
    # ====== ros_gz_bridge 起動 ======
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            "/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose"
        ],
        parameters=[{'config_file': bridge_yaml}],
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
    
    robot_description = Command(['xacro ', urdf_xacro, ' namespace:=', "", ' use_sim_time:=', use_sim_time])
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )
    
    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'params_file': params_file}.items()
    )
    
    gui_node = Node(
            package='dwpp_test_simulation',
            executable='follow_path_test_gui.py',
            name='follow_path_gui',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': use_sim_time,
                'nav2_params_file': params_file,
            }]
        )

    # ====== LaunchDescription ======
    ld = LaunchDescription()
    ld.add_action(declare_namespace)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_world)
    ld.add_action(declare_params)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_robot_sdf)
    ld.add_action(declare_yaml)
    ld.add_action(declare_rviz)
    ld.add_action(declare_urdfx)

    ld.add_action(gz_sim)
    ld.add_action(spawn_tb3)
    ld.add_action(bridge)
    ld.add_action(rviz)
    ld.add_action(static_map_to_odom)
    ld.add_action(nav2_navigation)
    ld.add_action(robot_state_publisher)
    ld.add_action(gui_node)

    return ld
