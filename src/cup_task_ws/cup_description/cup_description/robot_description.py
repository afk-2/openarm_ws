"""
Build the openarm_cup robot description.

Shared by the Gazebo launch and the MoveIt launch so robot_state_publisher,
Gazebo and move_group all use the exact same URDF.
"""
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import xacro

OPENARM_PREFIX = 'package://openarm_description/'
MIRRORED_PREFIX = 'package://cup_description/meshes/openarm_mirrored/'


def use_mirrored_collision_meshes(urdf):
    """
    Swap OpenArm's negatively scaled collision meshes for pre-mirrored copies.

    The left arm reuses the right arm's meshes with scale="1 -1 1", which
    DART (Gazebo physics) aborts on. cup_description installs y-mirrored
    copies of every collision STL (scripts/mirror_meshes.py). Visuals keep
    the negative scale; renderers handle it.
    """
    root = ET.fromstring(urdf)
    for mesh in root.iterfind('link/collision/geometry/mesh'):
        scale = [float(v) for v in mesh.get('scale', '1 1 1').split()]
        filename = mesh.get('filename')
        if scale[1] >= 0.0 or scale[0] < 0.0 or scale[2] < 0.0:
            continue
        if not filename.startswith(OPENARM_PREFIX):
            raise RuntimeError(f'No mirrored copy for negatively scaled mesh {filename}')
        mesh.set('filename', MIRRORED_PREFIX + filename[len(OPENARM_PREFIX):])
        mesh.set('scale', ' '.join(str(abs(v)) for v in scale))
    return ET.tostring(root, encoding='unicode')


def openarm_cup_urdf(mappings=None):
    """Process openarm_cup.urdf.xacro with the given xacro args."""
    xacro_file = os.path.join(
        get_package_share_directory('cup_description'), 'urdf', 'openarm_cup.urdf.xacro')
    urdf = xacro.process_file(xacro_file, mappings=mappings or {}).toxml()
    return use_mirrored_collision_meshes(urdf)
