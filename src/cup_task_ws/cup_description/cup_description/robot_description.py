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
FINGER_LIMIT_SLACK = 0.02  # rad, simulation only (see free_finger_end_stops)


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


def free_finger_end_stops(urdf, slack=FINGER_LIMIT_SLACK):
    """
    SIMULATION ONLY: move the finger joints' closed end stop `slack` rad past 0.

    In Gazebo (DART) a position-driven joint that sits exactly on its limit
    stays stuck there. OpenArm's fingers start on theirs (closed = 0 = upper
    limit): on a fresh simulation the second finger did not open with the
    first until something knocked it, and after a full close neither finger
    opened again. With the stop slightly past 0 the fingers never rest on it.
    Commands still stop at 0.
    """
    root = ET.fromstring(urdf)
    for joint in root.iterfind('joint'):
        limit = joint.find('limit')
        if 'finger_joint' in joint.get('name', '') and limit is not None:
            limit.set('upper', str(float(limit.get('upper')) + slack))
    return ET.tostring(root, encoding='unicode')


def openarm_cup_urdf(mappings=None):
    """Process openarm_cup.urdf.xacro with the given xacro args."""
    xacro_file = os.path.join(
        get_package_share_directory('cup_description'), 'urdf', 'openarm_cup.urdf.xacro')
    urdf = xacro.process_file(xacro_file, mappings=mappings or {}).toxml()
    urdf = use_mirrored_collision_meshes(urdf)
    if str((mappings or {}).get('use_gazebo', 'false')).lower() == 'true':
        urdf = free_finger_end_stops(urdf)
    return urdf
