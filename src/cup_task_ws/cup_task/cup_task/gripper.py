"""
Config-driven gripper: open/close by jaw opening in metres.

Everything gripper-specific (joint names, controller, opening <-> joint
table, finger dimensions) comes from the gripper config file, so the task
works unchanged with another gripper.
"""
import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from cup_task.grasps import GripperGeometry

PARAMS = {
    'arm_group': 'right_arm',
    'ee_link': '',
    'tool_axis': [0.0, 0.0, -1.0],
    'closing_axis': [0.0, 1.0, 0.0],
    'touch_links': [''],
    'controller_action': '',
    'joint_names': [''],
    'max_opening': 0.0,
    'opening_table': [0.0],
    'finger_length': 0.0,
    'fingertip_length': 0.0,
    'finger_width': 0.0,
    'close_duration': 1.5,
}


class Gripper:

    def __init__(self, node, action_client_factory):
        p = {name: node.declare_parameter(f'gripper.{name}', default).value
             for name, default in PARAMS.items()}
        missing = [k for k in ('ee_link', 'controller_action', 'max_opening', 'finger_length')
                   if not p[k]]
        if missing:
            raise ValueError(f'gripper config is missing: {missing}')
        self.arm_group = p['arm_group']
        self.ee_link = p['ee_link']
        self.touch_links = list(p['touch_links'])
        self.joint_names = list(p['joint_names'])
        self.max_opening = p['max_opening']
        self.close_duration = p['close_duration']
        table = np.array(p['opening_table'], dtype=float)
        cols = 1 + len(self.joint_names)
        if table.size % cols or table.size < 2 * cols:
            raise ValueError(f'gripper.opening_table needs rows of {cols} numbers '
                             '(opening, then one value per joint)')
        self.table = table.reshape(-1, cols)
        self.geometry = GripperGeometry(
            np.array(p['tool_axis'], dtype=float), np.array(p['closing_axis'], dtype=float),
            p['finger_length'], p['fingertip_length'] or p['finger_length'],
            p['finger_width'], self.max_opening)
        self.client = action_client_factory(FollowJointTrajectory, p['controller_action'])

    def joints_for(self, opening):
        """Joint positions for a jaw opening (m), capped at max_opening."""
        opening = min(max(opening, 0.0), self.max_opening)
        return [float(np.interp(opening, self.table[:, 0], self.table[:, 1 + i]))
                for i in range(len(self.joint_names))]

    def opening_from(self, joint_positions):
        """Jaw opening (m) from the first gripper joint's position."""
        col = self.table[:, 1]
        order = np.argsort(col)
        return float(np.interp(joint_positions[0], col[order], self.table[order, 0]))

    def goal(self, opening):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        sec = float(self.close_duration)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=self.joints_for(opening),
            time_from_start=Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9)))]
        return goal
