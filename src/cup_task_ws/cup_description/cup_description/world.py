"""
The Gazebo world for a given robot mounting height.

The robot stands on a pedestal; its base (the "world" frame of everything
ROS-side) is `spawn_z` above the floor. The table stays where it is (top at
0.75 m), so raising the robot means a taller pedestal. Shared by the Gazebo
launch and the MoveIt launch, so the simulated pedestal and the planning
scene obstacle always match the spawn height.
"""
import os
import tempfile
import xml.etree.ElementTree as ET

PEDESTAL = 'robot_pedestal'


def world_for_mount(world_file, spawn_z):
    """
    Path to `world_file` with the pedestal resized to `spawn_z` (m).

    Returns world_file itself when the pedestal already has that height;
    otherwise writes a copy to the temp directory (same name for the same
    height, so Gazebo and MoveIt read the same file).
    """
    spawn_z = float(spawn_z)
    tree = ET.parse(world_file)
    world = tree.getroot().find('world')
    model = next((m for m in world.findall('model') if m.get('name') == PEDESTAL), None)
    if model is None:
        raise RuntimeError(f'{world_file} has no "{PEDESTAL}" model')
    sizes = model.findall('link/collision/geometry/box/size') + \
        model.findall('link/visual/geometry/box/size')
    if all(abs(float(s.text.split()[2]) - spawn_z) < 1e-6 for s in sizes):
        return world_file
    for size in sizes:
        x, y, _ = size.text.split()
        size.text = f'{x} {y} {spawn_z}'
    pose = model.find('pose')
    x, y, _, roll, pitch, yaw = pose.text.split()
    pose.text = f'{x} {y} {spawn_z / 2} {roll} {pitch} {yaw}'  # box origin is its centre
    name = os.path.splitext(os.path.basename(world_file))[0]
    out = os.path.join(tempfile.gettempdir(), f'{name}_mount_{spawn_z:.3f}.sdf')
    tmp = f'{out}.{os.getpid()}'
    tree.write(tmp, xml_declaration=True, encoding='unicode')
    os.replace(tmp, out)  # atomic: Gazebo and MoveIt may write it at the same time
    return out
