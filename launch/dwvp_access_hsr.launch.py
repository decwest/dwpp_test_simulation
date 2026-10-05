"""Fixed-path Access experiments: explicit controller -> smoother -> HSR chain."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    package = get_package_share_directory('dwpp_test_simulation')
    hsr = get_package_share_directory('ytlab2_hsr_modules')
    params = LaunchConfiguration('params_file')
    sim_time = LaunchConfiguration('use_sim_time')
    configured = RewrittenYaml(
        source_file=params, root_key='',
        param_rewrites={'use_sim_time': sim_time, 'yaml_filename': LaunchConfiguration('map')},
        convert_types=True)
    nodes = [
        DeclareLaunchArgument('params_file', description='Frozen per-condition parameters selected by SESSION + TRIAL'),
        DeclareLaunchArgument('map', default_value=os.path.join(hsr, 'maps', 'dwvp_exp', 'map.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Show map, laser scan, frozen reference, and initial-pose tool'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(package, 'rviz', 'dwvp_access.rviz')),
        Node(package='nav2_map_server', executable='map_server', name='map_server', parameters=[configured], output='screen'),
        Node(package='nav2_amcl', executable='amcl', name='amcl', parameters=[configured], output='screen'),
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             parameters=[configured], remappings=[('cmd_vel', '/cmd_vel_nav')], output='screen'),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             parameters=[configured], remappings=[('cmd_vel', '/cmd_vel_nav'),
                                                  ('cmd_vel_smoothed', '/omni_base_controller/cmd_vel')], output='screen'),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_dwvp_access',
             parameters=[{'use_sim_time': sim_time, 'autostart': LaunchConfiguration('autostart'),
                          'node_names': ['map_server', 'amcl', 'controller_server', 'velocity_smoother']}], output='screen'),
        Node(package='rviz2', executable='rviz2', name='dwvp_access_rviz',
             condition=IfCondition(LaunchConfiguration('use_rviz')),
             arguments=['-d', LaunchConfiguration('rviz_config')],
             parameters=[{'use_sim_time': sim_time}], output='screen'),
    ]
    return LaunchDescription(nodes)
