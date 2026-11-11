"""
MoveIt 2 for openarm_cup (OpenArm v2.0 + D435 mast).

Normally included by cup_bringup/sim.launch.py moveit:=true, after the
Gazebo controllers are up. Standalone (sim already running):

    ros2 launch cup_moveit_config move_group.launch.py

Reused from openarm_bimanual_moveit_config (openarm_v2.0): SRDF (wrapped in
config/openarm_cup.srdf.xacro), joint_limits.yaml; kinematics.yaml is cup_moveit_config's own
(KDL for the arms, pick_ik for the right_arm_ik grasp-search group).
Our own: robot description (cup_description), controllers, planning scene
obstacles taken from the Gazebo world, RViz config.
"""
import os

from ament_index_python.packages import get_package_share_directory
from cup_description.robot_description import openarm_cup_urdf
from cup_description.world import world_for_mount
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

CAMERA_ARGS = ('camera_mast_height', 'camera_forward_offset', 'camera_tilt')


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    use_sim_time = arg('use_sim_time') == 'true'
    upstream_config = os.path.join(
        get_package_share_directory('openarm_bimanual_moveit_config'), 'config', 'openarm_v2.0')
    description_mappings = {
        # The Gazebo variant only differs in ros2_control/<gazebo> tags, which
        # MoveIt ignores; keep it identical to robot_state_publisher's anyway.
        'use_gazebo': str(use_sim_time).lower(),
        **{name: arg(name) for name in CAMERA_ARGS},
    }

    moveit_config = (
        MoveItConfigsBuilder('openarm_cup', package_name='cup_moveit_config')
        .robot_description(
            file_path=os.path.join(get_package_share_directory('cup_description'),
                                   'urdf', 'openarm_cup.urdf.xacro'),
            mappings=description_mappings)
        .robot_description_semantic(file_path='config/openarm_cup.srdf.xacro')
        # Own kinematics.yaml: KDL for the arms, pick_ik for the right_arm_ik group.
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path=os.path.join(upstream_config, 'joint_limits.yaml'))
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .planning_scene_monitor(publish_robot_description=True,
                                publish_robot_description_semantic=True)
        .to_moveit_configs()
    )
    # Same URDF as robot_state_publisher/Gazebo, incl. the mirrored collision meshes.
    moveit_config.robot_description = {
        'robot_description': openarm_cup_urdf(description_mappings)}

    sim_time = {'use_sim_time': use_sim_time}

    params = moveit_config.to_dict()
    if arg('grasp_ik_threads'):  # '' = keep the value in kinematics.yaml
        params['robot_description_kinematics']['right_arm_ik']['memetic_num_threads'] = \
            int(arg('grasp_ik_threads'))

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[params, sim_time],
    )

    models = ['table', 'robot_pedestal'] + (['ceramic_cup'] if arg('add_cup') == 'true' else [])
    world_obstacles = Node(
        package='cup_moveit_config',
        executable='add_world_obstacles.py',
        output='screen',
        parameters=[{
            # Pedestal resized to the spawn height, as in Gazebo.
            'world_file': world_for_mount(arg('world'), arg('spawn_z')),
            'models': models,
            'spawn_xyz': [0.0, 0.0, float(arg('spawn_z'))],
        }, sim_time],
    )

    nodes = [move_group, world_obstacles]
    if arg('rviz') == 'true':
        nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            output='log',
            arguments=['-d', os.path.join(get_package_share_directory('cup_moveit_config'),
                                          'config', 'moveit.rviz')],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                sim_time,
            ],
        ))
    return nodes


def generate_launch_description():
    description_share = get_package_share_directory('cup_description')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Start RViz with the MotionPlanning plugin.'),
        DeclareLaunchArgument(
            'world', default_value=os.path.join(description_share, 'worlds', 'cup_table.sdf'),
            description='Gazebo world to take the planning scene obstacles from.'),
        DeclareLaunchArgument('spawn_z', default_value='0.40',
                              description='Robot mounting height above the floor (m); '
                                          'the pedestal obstacle is resized to match.'),
        DeclareLaunchArgument(
            'grasp_ik_threads', default_value='',
            description='Threads of the grasp-search IK solver (pick_ik, right_arm_ik '
                        'group). Empty = kinematics.yaml (4). sim.launch.py passes 1 so '
                        'Gazebo keeps enough CPU on a laptop.'),
        DeclareLaunchArgument(
            'add_cup', default_value='true',
            description='Also add the cup from the world file (ground truth, a stand-in '
                        'until perception publishes it).'),
        # Defaults must match openarm_cup.urdf.xacro.
        DeclareLaunchArgument('camera_mast_height', default_value='0.30'),
        DeclareLaunchArgument('camera_forward_offset', default_value='0.10'),
        DeclareLaunchArgument('camera_tilt', default_value='1.10'),
        OpaqueFunction(function=launch_setup),
    ])
