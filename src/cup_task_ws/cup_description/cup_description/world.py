"""
The Gazebo world for a given robot mounting height.

The robot stands on a pedestal; its base (the "world" frame of everything
ROS-side) is `spawn_z` above the floor. The table stays where it is (top at
0.75 m), so raising the robot means a taller pedestal. Shared by the Gazebo
launch and the MoveIt launch, so the simulated pedestal and the planning
scene obstacle always match the spawn height. The Gazebo launch also sets
the physics step this way (simulation speed setting).
"""
import os
import tempfile
import xml.etree.ElementTree as ET

PEDESTAL = 'robot_pedestal'


def world_for_mount(world_file, spawn_z, physics_step=None):
    """
    Path to `world_file` with the pedestal resized to `spawn_z` (m) and,
    if given, the physics step set to `physics_step` (s).

    Returns world_file itself when nothing needs changing; otherwise writes a
    copy to the temp directory (same name for the same values, so Gazebo and
    MoveIt read the same pedestal).
    """
    spawn_z = float(spawn_z)
    tree = ET.parse(world_file)
    world = tree.getroot().find('world')
    model = next((m for m in world.findall('model') if m.get('name') == PEDESTAL), None)
    if model is None:
        raise RuntimeError(f'{world_file} has no "{PEDESTAL}" model')
    sizes = model.findall('link/collision/geometry/box/size') + \
        model.findall('link/visual/geometry/box/size')
    step = world.find('physics/max_step_size')
    same_height = all(abs(float(s.text.split()[2]) - spawn_z) < 1e-6 for s in sizes)
    same_step = physics_step is None or abs(float(step.text) - float(physics_step)) < 1e-9
    if same_height and same_step:
        return world_file
    for size in sizes:
        x, y, _ = size.text.split()
        size.text = f'{x} {y} {spawn_z}'
    pose = model.find('pose')
    x, y, _, roll, pitch, yaw = pose.text.split()
    pose.text = f'{x} {y} {spawn_z / 2} {roll} {pitch} {yaw}'  # box origin is its centre
    name = os.path.splitext(os.path.basename(world_file))[0]
    if not same_step:
        step.text = str(float(physics_step))
        name += f'_step_{float(physics_step):g}'
    out = os.path.join(tempfile.gettempdir(), f'{name}_mount_{spawn_z:.3f}.sdf')
    tmp = f'{out}.{os.getpid()}'
    tree.write(tmp, xml_declaration=True, encoding='unicode')
    os.replace(tmp, out)  # atomic: Gazebo and MoveIt may write it at the same time
    return out
