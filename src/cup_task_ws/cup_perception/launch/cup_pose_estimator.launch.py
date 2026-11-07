"""
Run the cup pose estimator (needs the sim, or the real D435 driver, running).

    ros2 launch cup_perception cup_pose_estimator.launch.py [rviz:=true] [depth_noise:=true]

depth_noise:=true (simulation only) starts d435_depth_noise, which adds
D435-style errors to Gazebo's perfect depth, and points the estimator at its
output topic.

cup_pose_topic:=/cup_pose publishes the target cup's pose there instead of
/cup_pose_estimate, so pick_place can use it (handle_visible stays on
/cup_pose_estimate/handle_visible).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('cup_perception')
    use_sim_time = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'),
                                                   value_type=bool)}
    remap = [('/cup_pose_estimate', LaunchConfiguration('cup_pose_topic'))]
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(
            share, 'config', 'cup_pose_estimator.yaml'), description='Thresholds file.'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Open RViz with the perception displays.'),
        DeclareLaunchArgument('depth_noise', default_value='false',
                              description='SIMULATION ONLY: add D435-style depth errors.'),
        DeclareLaunchArgument('cup_pose_topic', default_value='/cup_pose_estimate',
                              description='Topic for the target cup pose.'),
        DeclareLaunchArgument('noise_config', default_value=os.path.join(
            share, 'config', 'd435_depth_noise.yaml'), description='Depth noise settings.'),
        Node(package='cup_perception', executable='cup_pose_estimator', output='screen',
             parameters=[LaunchConfiguration('config'), use_sim_time],
             remappings=remap,
             condition=UnlessCondition(LaunchConfiguration('depth_noise'))),
        Node(package='cup_perception', executable='cup_pose_estimator', output='screen',
             parameters=[LaunchConfiguration('config'), use_sim_time,
                         {'depth_topic': '/camera/camera/aligned_depth_to_color/image_noisy'}],
             remappings=remap,
             condition=IfCondition(LaunchConfiguration('depth_noise'))),
        Node(package='cup_perception', executable='d435_depth_noise', output='screen',
             parameters=[LaunchConfiguration('noise_config'), use_sim_time],
             condition=IfCondition(LaunchConfiguration('depth_noise'))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(share, 'rviz', 'perception.rviz')],
             parameters=[use_sim_time], condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
