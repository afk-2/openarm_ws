"""
Run the cup pick-and-place.

Needs the sim with MoveIt already running:
`ros2 launch cup_bringup sim.launch.py moveit:=true`.

    ros2 launch cup_task pick_place.launch.py
        [cup_pose_source:=gazebo|none] [gripper_config:=<file>] [use_sim_attach:=true]

cup_pose_source:=gazebo starts cup_perception's Gazebo ground-truth source.
Use cup_pose_source:=none when another node (e.g. your perception) publishes
/cup_pose.

use_sim_attach (default true here, because this launch file is for the
simulation) locks the cup to the hand in Gazebo while it is held; see
cup_task/sim_attach.py. On the real robot: use_sim_attach:=false.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('cup_task'), 'config')
    return LaunchDescription([
        DeclareLaunchArgument('cup_pose_source', default_value='gazebo',
                              choices=['gazebo', 'none'],
                              description='Who publishes /cup_pose.'),
        DeclareLaunchArgument('gripper_config',
                              default_value=os.path.join(config, 'gripper_openarm_pinch.yaml'),
                              description='Gripper description (joints, max opening, fingers).'),
        DeclareLaunchArgument('task_config',
                              default_value=os.path.join(config, 'pick_place.yaml'),
                              description='Task settings (place spot, distances, speeds).'),
        DeclareLaunchArgument('use_sim_attach', default_value='true',
                              description='SIMULATION ONLY: lock the cup to the hand in '
                                          'Gazebo while held (needs sim.launch.py '
                                          'sim_cup_attach:=true).'),

        Node(
            package='cup_perception',
            executable='gazebo_cup_pose',
            output='screen',
            # spawn_xyz defaults to sim.launch.py's robot spawn position.
            parameters=[{'use_sim_time': True}],
            condition=LaunchConfigurationEquals('cup_pose_source', 'gazebo'),
        ),
        Node(
            package='cup_task',
            executable='pick_place',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('task_config'),
                        LaunchConfiguration('gripper_config'),
                        {'use_sim_time': True,
                         'use_sim_attach': ParameterValue(
                             LaunchConfiguration('use_sim_attach'), value_type=bool)}],
        ),
    ])
