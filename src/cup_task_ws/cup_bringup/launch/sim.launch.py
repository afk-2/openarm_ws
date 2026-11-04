"""
OpenArm + D435 + ceramic cup in Gazebo Fortress.

    ros2 launch cup_bringup sim.launch.py [moveit:=true] [rviz:=true] [gui:=true]

Starts Gazebo with the cup_table world, spawns OpenArm on its pedestal,
bridges /clock and the camera topics, starts the controllers and RViz.
With moveit:=true it also starts move_group (cup_moveit_config) once the
controllers are up, and RViz opens with the MotionPlanning plugin instead.
Every ROS node runs with use_sim_time:=true.
"""
import os

from ament_index_python.packages import get_package_share_directory
from cup_description.robot_description import openarm_cup_urdf
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

CAMERA_ARGS = ('camera_mast_height', 'camera_forward_offset', 'camera_tilt')


def running_on_wsl():
    try:
        with open('/proc/version') as f:
            return 'microsoft' in f.read().lower()
    except OSError:
        return False


def generate_launch_description():
    description_share = get_package_share_directory('cup_description')
    bringup_share = get_package_share_directory('cup_bringup')

    use_sim_time = {'use_sim_time': True}
    controllers_file = os.path.join(bringup_share, 'config', 'sim_controllers.yaml')
    world = LaunchConfiguration('world')

    declared_arguments = [
        DeclareLaunchArgument(
            'world', default_value=os.path.join(description_share, 'worlds', 'cup_table.sdf'),
            description='Gazebo world file.'),
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo GUI.'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz.'),
        DeclareLaunchArgument(
            'moveit', default_value='false',
            description='Also start move_group and RViz with the MotionPlanning plugin.'),
        DeclareLaunchArgument(
            'render_engine', default_value='ogre2', choices=['ogre', 'ogre2'],
            description='Gazebo render engine.'),
        DeclareLaunchArgument(
            'software_gl', default_value=str(running_on_wsl()).lower(),
            description='Render Gazebo with Mesa llvmpipe. ogre2 needs OpenGL >= 4.3, '
                        "which WSL2's D3D12 driver lacks (ogre renders blank sensor "
                        'images there), so this defaults to true on WSL.'),
        DeclareLaunchArgument(
            'sim_cup_attach', default_value='true',
            description='SIMULATION-ONLY shortcut: let cup_task lock the cup to the hand '
                        'while held (Gazebo point contacts let a pinched cup pivot).'),
        DeclareLaunchArgument(
            'spawn_z', default_value='0.40',
            description='Height of the robot base (top of the pedestal in the world).'),
        # Defaults must match openarm_cup.urdf.xacro.
        DeclareLaunchArgument('camera_mast_height', default_value='0.30',
                              description='Mast height above the body top (m).'),
        DeclareLaunchArgument('camera_forward_offset', default_value='0.10',
                              description='Bracket reach in front of the mast (m).'),
        DeclareLaunchArgument('camera_tilt', default_value='1.10',
                              description='Downward camera pitch (rad).'),
    ]

    # Let Gazebo resolve package:// and model:// URIs (OpenArm, D435 and cup meshes).
    resource_paths = [
        AppendEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH',
            os.path.dirname(get_package_share_directory(pkg)))
        for pkg in ('openarm_description', 'realsense2_description', 'cup_description')
    ] + [
        AppendEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH', os.path.join(description_share, 'models')),
    ]

    def robot_state_publisher(context):
        urdf = openarm_cup_urdf({
            'use_gazebo': 'true',
            'controllers_file': controllers_file,
            'sim_cup_attach': LaunchConfiguration('sim_cup_attach').perform(context),
            **{arg: LaunchConfiguration(arg).perform(context) for arg in CAMERA_ARGS},
        })
        return [Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': urdf}, use_sim_time],
        )]

    def gazebo(context):
        cmd = ['ign', 'gazebo', '-r', '-v', '3',
               '--render-engine', LaunchConfiguration('render_engine').perform(context),
               world.perform(context)]
        if LaunchConfiguration('gui').perform(context) != 'true':
            cmd.insert(2, '-s')
        # Same plugin lookup ros_gz_sim's gz_sim.launch.py sets up (finds ign_ros2_control).
        env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.pathsep.join(
            p for p in (os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
                        os.environ.get('LD_LIBRARY_PATH', '')) if p)}
        # Only Gazebo gets software GL; RViz keeps the GPU.
        if LaunchConfiguration('software_gl').perform(context) == 'true':
            env['LIBGL_ALWAYS_SOFTWARE'] = '1'
        process = ExecuteProcess(cmd=cmd, output='screen', additional_env=env)
        return [process, RegisterEventHandler(OnProcessExit(
            target_action=process, on_exit=[EmitEvent(event=Shutdown(reason='Gazebo exited'))]))]

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'openarm',
                   '-x', '0', '-y', '0', '-z', LaunchConfiguration('spawn_z')],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'config_file': os.path.join(bringup_share, 'config', 'gz_bridge.yaml')},
                    use_sim_time],
    )

    cloud_to_optical = Node(
        package='cup_bringup',
        executable='gz_cloud_to_optical.py',
        output='screen',
        remappings=[('points_in', '/camera/camera/depth/color/points_gz'),
                    ('points_out', '/camera/camera/depth/color/points')],
        parameters=[use_sim_time],
    )

    def spawner(*controllers):
        return Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[*controllers, '--controller-manager', '/controller_manager',
                       '--controller-manager-timeout', '60'],
            parameters=[use_sim_time],
        )

    joint_state_broadcaster = spawner('joint_state_broadcaster')

    # The DetachableJoint plugin attaches the cup as soon as the robot loads;
    # release it so the cup starts free on the table.
    release_cup = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--times', '3', '--rate', '2',
             '/cup_sim_attach/detach', 'std_msgs/msg/Empty', '{}'],
        output='log',
        condition=IfCondition(LaunchConfiguration('sim_cup_attach')),
    )
    arm_and_gripper_controllers = spawner(
        'left_joint_trajectory_controller', 'right_joint_trajectory_controller',
        'left_gripper_controller', 'right_gripper_controller')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='log',
        arguments=['-d', PathJoinSubstitution([bringup_share, 'rviz', 'sim.rviz'])],
        parameters=[use_sim_time],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('rviz'), "' == 'true' and '",
            LaunchConfiguration('moveit'), "' != 'true'"])),
    )

    # move_group + MoveIt RViz, using the same URDF args as robot_state_publisher.
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('cup_moveit_config'), 'launch', 'move_group.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'rviz': LaunchConfiguration('rviz'),
            'world': world,
            'spawn_z': LaunchConfiguration('spawn_z'),
            **{arg: LaunchConfiguration(arg) for arg in CAMERA_ARGS},
        }.items(),
        condition=IfCondition(LaunchConfiguration('moveit')),
    )

    return LaunchDescription(declared_arguments + resource_paths + [
        OpaqueFunction(function=gazebo),
        OpaqueFunction(function=robot_state_publisher),
        bridge,
        cloud_to_optical,
        spawn_robot,
        # The controller manager only exists once the robot is in Gazebo.
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot, on_exit=[joint_state_broadcaster, release_cup])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[arm_and_gripper_controllers])),
        # MoveIt needs joint states and the trajectory controllers to execute plans.
        RegisterEventHandler(OnProcessExit(
            target_action=arm_and_gripper_controllers, on_exit=[moveit])),
        rviz,
    ])
