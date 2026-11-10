"""
Offline reachability map for pick_place (MoveIt only, no Gazebo).

    ros2 launch cup_task reachability.launch.py spawn_z:=0.40 [args:="--step 0.05"]

Starts move_group (planning scene: table and pedestal for this mounting
height) and robot_state_publisher (for the camera pose), runs
cup_task/reachability_map.py, and shuts everything down when it finishes.
Results go to cup_task_ws/results/reachability_mount_<spawn_z>.{csv,png}.
"""
import os

from ament_index_python.packages import get_package_share_directory
from cup_description.robot_description import openarm_cup_urdf
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    spawn_z = LaunchConfiguration('spawn_z').perform(context)
    extra = LaunchConfiguration('args').perform(context).split()
    # For the camera pose only. The simulated-camera variant of the URDF has
    # the optical frames (on the real robot the camera driver adds them);
    # the geometry is identical.
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               output='log',
               parameters=[{'robot_description': openarm_cup_urdf({'use_gazebo': 'true'})}])
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('cup_moveit_config'), 'launch', 'move_group.launch.py')),
        launch_arguments={'use_sim_time': 'false', 'rviz': 'false', 'add_cup': 'false',
                          'spawn_z': spawn_z}.items())
    mapper = Node(package='cup_task', executable='reachability_map', output='screen',
                  emulate_tty=True, arguments=['--spawn-z', spawn_z] + extra)
    return [rsp, moveit, mapper, RegisterEventHandler(OnProcessExit(
        target_action=mapper, on_exit=[EmitEvent(event=Shutdown(reason='map done'))]))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('spawn_z', default_value='0.40',
                              description='Robot mounting height (m), as sim.launch.py.'),
        DeclareLaunchArgument('args', default_value='',
                              description='Extra arguments for reachability_map.'),
        OpaqueFunction(function=setup),
    ])
