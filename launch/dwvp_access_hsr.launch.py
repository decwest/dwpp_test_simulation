"""Fixed-path Access experiments: explicit controller -> smoother -> HSR chain."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    package = get_package_share_directory('dwpp_test_simulation')
    hsr = get_package_share_directory('ytlab2_hsr_modules')
    params = LaunchConfiguration('params_file')
    sim_time = LaunchConfiguration('use_sim_time')
    def stop_on_exit(event, context):
        if not context.is_shutdown:
            return [EmitEvent(event=Shutdown(reason='An experiment launch process exited'))]
        return []
    configured = RewrittenYaml(
        source_file=params, root_key='',
        param_rewrites={'use_sim_time': sim_time, 'yaml_filename': LaunchConfiguration('map')},
        convert_types=True)
    nodes = [
        DeclareLaunchArgument('params_file', description='Frozen per-condition parameters selected by SESSION + TRIAL'),
        DeclareLaunchArgument('map', default_value=os.path.join(hsr, 'maps', 'dwvp_exp', 'map.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('shutdown_on_exit', default_value='false',
                              description='Shut down this launch if any owned process exits'),
        RegisterEventHandler(OnProcessExit(on_exit=stop_on_exit),
                             condition=IfCondition(LaunchConfiguration('shutdown_on_exit'))),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('localization_only', default_value='false',
                              description='Start map/AMCL/RViz before freezing the measured start; no velocity publishers'),
        DeclareLaunchArgument('start_localization', default_value='true',
                              description='Set false when map/AMCL are already running via localize'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Show map, laser scan, frozen reference, and initial-pose tool'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(package, 'rviz', 'dwvp_access.rviz')),
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             condition=IfCondition(LaunchConfiguration('start_localization')), parameters=[configured], output='screen'),
        Node(package='nav2_amcl', executable='amcl', name='amcl',
             condition=IfCondition(LaunchConfiguration('start_localization')), parameters=[configured], output='screen'),
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             condition=UnlessCondition(LaunchConfiguration('localization_only')),
             parameters=[configured], remappings=[('cmd_vel', '/cmd_vel_nav')], output='screen'),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             condition=UnlessCondition(LaunchConfiguration('localization_only')),
             parameters=[configured], remappings=[('cmd_vel', '/cmd_vel_nav'),
                                                  ('cmd_vel_smoothed', '/omni_base_controller/cmd_vel')], output='screen'),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_dwvp_localization',
             condition=IfCondition(LaunchConfiguration('start_localization')),
             parameters=[{'use_sim_time': sim_time, 'autostart': LaunchConfiguration('autostart'),
                          'node_names': ['map_server', 'amcl']}], output='screen'),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_dwvp_access',
             condition=UnlessCondition(LaunchConfiguration('localization_only')),
             parameters=[{'use_sim_time': sim_time, 'autostart': LaunchConfiguration('autostart'),
                          'node_names': ['controller_server', 'velocity_smoother']}], output='screen'),
        Node(package='rviz2', executable='rviz2', name='dwvp_access_rviz',
             condition=IfCondition(LaunchConfiguration('use_rviz')),
             arguments=['-d', LaunchConfiguration('rviz_config')],
             parameters=[{'use_sim_time': sim_time}], output='screen'),
    ]
    return LaunchDescription(nodes)
