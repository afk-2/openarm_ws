"""
Grasp geometry for a body grasp of an upright cup (pure maths, no ROS).

A grasp is described by where the gripper's axes point in the world:
  tool     direction the fingers point (palm -> fingertips)
  closing  direction the fingers close along
  width    the remaining axis, along the finger width
Candidates come from every direction around the cup:
  - the hand approaches from a heading (any direction around the cup),
    pitched down by an angle: 0 deg = from the side, 90 deg = straight
    down from above
  - the closing axis is level and at right angles to the approach, so the
    fingers squeeze the cup body symmetrically through its centre
  - the fingers close at the requested height on the body, raised if needed
    to keep the finger corners off the table
Whether the fingers stay clear of the handle depends on the heading; the
task checks that with MoveIt (pick_place.handle_clearance).
The cup stays level because the hand orientation does not change while the
cup is held, whatever the pitch.
"""
from dataclasses import dataclass
import math

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Cup:
    outer_diameter: float
    height: float
    handle_radius: float
    handle_thickness: float
    handle_center_height: float

    @classmethod
    def from_config(cls, cfg):
        h = cfg['handle']
        return cls(cfg['outer_diameter'], cfg['height'], h['radius'], h['thickness'],
                   h['center_height'])

    @property
    def radius(self):
        return self.outer_diameter / 2


@dataclass
class GripperGeometry:
    tool_axis: np.ndarray      # in ee_link frame
    closing_axis: np.ndarray   # in ee_link frame
    finger_length: float       # ee_link -> pad centre along tool axis
    fingertip_length: float    # ee_link -> fingertip along tool axis
    finger_width: float
    max_opening: float


@dataclass
class Grasp:
    name: str
    rotation: np.ndarray       # ee_link orientation in the world (3x3)
    approach: np.ndarray       # unit vector the gripper moves along to reach the cup
    contact_height: float      # above the cup base
    heading_deg: float = 0.0   # horizontal approach direction in the world


def yaw_of(quat_xyzw):
    x, y, z, w = quat_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(quat_xyzw):
    """Angle between an object's z axis and vertical."""
    x, y, _, _ = quat_xyzw
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def quat_from_matrix(m):
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return x, y, z, w


def _rotation(gripper, tool, closing):
    """ee_link orientation that points the gripper's tool/closing axes as given."""
    width_ee = np.cross(gripper.tool_axis, gripper.closing_axis)
    ee_axes = np.column_stack([gripper.tool_axis, gripper.closing_axis, width_ee])
    world_axes = np.column_stack([tool, closing, np.cross(tool, closing)])
    return world_axes @ ee_axes.T


def candidates(cup, gripper, height_fraction, table_clearance, pitches_deg, headings_deg):
    """
    Grasp candidates for every approach heading (world deg, the horizontal
    direction the hand moves in to reach the cup) and pitch.
    """
    tip_extra = gripper.fingertip_length - gripper.finger_length
    out = []
    for heading in headings_deg:
        along = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
        label = f'heading {heading:.0f} deg'
        for pitch_deg in pitches_deg:
            pitch = math.radians(pitch_deg)
            tool = math.cos(pitch) * along - math.sin(pitch) * UP
            for wrist, closing in (('', np.cross(UP, along)),
                                   (', wrist flipped', np.cross(along, UP))):
                width = np.cross(tool, closing)
                # Lowest finger corner: pad centre +- half the finger width, plus the
                # fingertip beyond the pad. Keep it table_clearance above the table.
                lowest = gripper.finger_width / 2 * abs(width[2]) - tip_extra * tool[2]
                height = max(height_fraction * cup.height, table_clearance + lowest)
                if height > 0.85 * cup.height:
                    continue  # fingers would only catch the rim
                out.append(Grasp(f'{label}, pitch {pitch_deg:.0f} deg{wrist}',
                                 _rotation(gripper, tool, closing), tool, height, heading))
    return out


def ee_position(grasp, cup_base, gripper):
    """ee_link position that puts the finger pads on the cup at the grasp height."""
    contact = cup_base + UP * grasp.contact_height
    tool = grasp.rotation @ gripper.tool_axis
    return contact - tool * gripper.finger_length


def check_fits(cup, gripper):
    """Raise ValueError if this gripper cannot body-grasp this cup."""
    if cup.outer_diameter >= gripper.max_opening:
        raise ValueError(f'cup is {cup.outer_diameter * 1000:.0f} mm wide but the gripper only '
                         f'opens {gripper.max_opening * 1000:.0f} mm')
    return (gripper.max_opening - cup.outer_diameter) / 2


def angle_to_closing_line(direction_yaw, grasp):
    """
    Angle (deg, 0..90) between a horizontal direction and the line the
    fingers close along: 0 = pointing straight at a finger.
    """
    c = math.radians(grasp.heading_deg + 90.0)
    d = abs(math.cos(direction_yaw - c))
    return math.degrees(math.acos(min(1.0, d)))
