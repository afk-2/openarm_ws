"""
Run the cup pick-and-place.

Needs the sim with MoveIt already running:
`ros2 launch cup_bringup sim.launch.py moveit:=true`.

    ros2 launch cup_task pick_place.launch.py
        [cup_pose_source:=gazebo|none] [gripper_config:=<file>] [use_sim_attach:=true]

cup_pose_source:=gazebo starts cup_perception's Gazebo ground-truth source.
Use cup_pose_source:=none when another node publishes /cup_pose. For the
camera estimator, also pass pose_from_perception:=true (it cannot see a held
cup, and it reports whether it saw the handle):

    ros2 launch cup_perception cup_pose_estimator.launch.py cup_pose_topic:=/cup_pose
    ros2 launch cup_task pick_place.launch.py cup_pose_source:=none pose_from_perception:=true

use_sim_attach (default true here, because this launch file is for the
simulation) locks the cup to the hand in Gazebo while it is held; see
cup_task/sim_attach.py. On the real robot: use_sim_attach:=false.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def gazebo_pose_source(context):
    """Gazebo ground-truth cup pose, shifted by the robot's mounting height."""
    if LaunchConfiguration('cup_pose_source').perform(context) != 'gazebo':
        return []
    spawn_z = float(LaunchConfiguration('spawn_z').perform(context))
    return [Node(
        package='cup_perception',
        executable='gazebo_cup_pose',
        output='screen',
        parameters=[{'use_sim_time': True, 'spawn_xyz': [0.0, 0.0, spawn_z]}],
    )]


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
        DeclareLaunchArgument('spawn_z', default_value='0.40',
                              description='Robot mounting height (as sim.launch.py); used '
                                          'by the Gazebo ground-truth pose source.'),
        DeclareLaunchArgument('pose_from_perception', default_value='false',
                              description='/cup_pose comes from cup_pose_estimator: skip '
                                          'in-hand checks, use its handle_visible flag.'),
        DeclareLaunchArgument('use_sim_attach', default_value='true',
                              description='SIMULATION ONLY: lock the cup to the hand in '
                                          'Gazebo while held (needs sim.launch.py '
                                          'sim_cup_attach:=true).'),

        OpaqueFunction(function=gazebo_pose_source),
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
            condition=UnlessCondition(LaunchConfiguration('pose_from_perception')),
        ),
        # Same node with the camera estimator's settings.
        Node(
            package='cup_task',
            executable='pick_place',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('task_config'),
                        LaunchConfiguration('gripper_config'),
                        {'use_sim_time': True,
                         'use_sim_attach': ParameterValue(
                             LaunchConfiguration('use_sim_attach'), value_type=bool),
                         'pose_source_sees_held_cup': False,
                         'handle_visible_topic': '/cup_pose_estimate/handle_visible'}],
            condition=IfCondition(LaunchConfiguration('pose_from_perception')),
        ),
    ])
