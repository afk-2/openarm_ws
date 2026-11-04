"""
Run the cup pose estimator (needs the sim, or the real D435 driver, running).

    ros2 launch cup_perception cup_pose_estimator.launch.py [rviz:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('cup_perception')
    use_sim_time = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'),
                                                   value_type=bool)}
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(
            share, 'config', 'cup_pose_estimator.yaml'), description='Thresholds file.'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Open RViz with the perception displays.'),
        Node(package='cup_perception', executable='cup_pose_estimator', output='screen',
             parameters=[LaunchConfiguration('config'), use_sim_time]),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(share, 'rviz', 'perception.rviz')],
             parameters=[use_sim_time], condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
