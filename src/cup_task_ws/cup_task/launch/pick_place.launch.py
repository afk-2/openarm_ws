"""
Run the cup pick-and-place.

Needs the sim with MoveIt already running:
`ros2 launch cup_bringup sim.launch.py moveit:=true`.

    ros2 launch cup_task pick_place.launch.py
        [cup_pose_source:=gazebo|none] [gripper_config:=<file>] [sim:=false]

cup_pose_source:=gazebo starts cup_perception's Gazebo ground-truth source.
Use cup_pose_source:=none when another node publishes /cup_pose. For the
camera estimator, also pass pose_from_perception:=true (it cannot see a held
cup, and it reports whether it saw the handle):

    ros2 launch cup_perception cup_pose_estimator.launch.py cup_pose_topic:=/cup_pose
    ros2 launch cup_task pick_place.launch.py cup_pose_source:=none pose_from_perception:=true

sim (default true) loads config/pick_place_sim.yaml on top of the task
settings: the simulation-only shortcuts (cup locked to the hand in Gazebo
while held, gripper closed to a width instead of until it stalls, ...). On
the real robot: sim:=false, which leaves everything as in pick_place.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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


def pick_place_node(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    sim = arg('sim') == 'true'
    params = [arg('task_config'), arg('gripper_config')]
    if sim:
        params.append(arg('sim_config'))  # simulation-only settings, on top
    overrides = {'use_sim_time': sim}
    if arg('use_sim_attach'):
        overrides['use_sim_attach'] = arg('use_sim_attach') == 'true'
    if arg('pose_from_perception') == 'true':
        # The camera estimator cannot see a held cup, and says whether it saw the handle.
        overrides.update({'pose_source_sees_held_cup': False,
                          'handle_visible_topic': '/cup_pose_estimate/handle_visible'})
    return [Node(package='cup_task', executable='pick_place', output='screen',
                 emulate_tty=True, parameters=params + [overrides])]


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
        DeclareLaunchArgument('sim', default_value='true', choices=['true', 'false'],
                              description='Load the simulation-only settings '
                                          '(config/pick_place_sim.yaml). false = real robot.'),
        DeclareLaunchArgument('sim_config',
                              default_value=os.path.join(config, 'pick_place_sim.yaml'),
                              description='Simulation-only settings, used with sim:=true.'),
        DeclareLaunchArgument('use_sim_attach', default_value='',
                              description='SIMULATION ONLY: override use_sim_attach '
                                          '(true/false; empty = as the settings files say).'),

        OpaqueFunction(function=gazebo_pose_source),
        OpaqueFunction(function=pick_place_node),
    ])
