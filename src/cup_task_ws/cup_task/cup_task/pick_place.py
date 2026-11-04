"""
Pick-and-place of a (possibly full) cup with a body grasp, keeping it level.

Sequence, logged step by step:
  lift clear of the table -> open -> move above the cup -> descend -> close
  -> lift -> move to the place pose -> lower -> open -> retreat -> home

Inputs (all swappable):
  /cup_pose        geometry_msgs/PoseStamped from any cup pose source
                   (base centre of the cup, x axis towards the handle)
  cup config       cup dimensions (cup_description/config/cup.yaml by default)
  gripper config   joint names, max opening, finger size (gripper_*.yaml)

Before moving, every pose of the chosen grasp (approach, grasp, lift, carry,
place, retreat) is checked with IK, so the robot does not start a motion it
cannot finish. The hand orientation never changes while the cup is held, so
the cup stays level.
"""
import math
import os
import threading
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from cup_task import grasps
from cup_task.gripper import Gripper
from cup_task.moveit_client import MoveItClient, TaskError, pose_msg, wait

CUP_ID = 'ceramic_cup'  # same id as the world obstacle, so it gets replaced
UP = grasps.UP


class PickPlace(Node):

    def __init__(self):
        super().__init__('pick_place')
        self.cb = ReentrantCallbackGroup()
        p = self.declare_parameter
        self.cup_pose_topic = p('cup_pose_topic', '/cup_pose').value
        cup_config = p('cup_config', '').value or os.path.join(
            get_package_share_directory('cup_description'), 'config', 'cup.yaml')
        self.frame = p('planning_frame', 'world').value
        self.home_pose = p('home_pose', 'home').value
        self.lifted_pose = p('lifted_pose', 'lifted').value
        self.place_xy = p('place_xy', [0.30, -0.32]).value
        self.height_fraction = p('grasp_height_fraction', 0.5).value
        self.table_clearance = p('table_clearance', 0.008).value
        self.pitches = p('grasp_pitches_deg', [45.0, 30.0, 60.0, 15.0, 75.0, 0.0, 90.0]).value
        self.approach_distance = p('approach_distance', 0.08).value
        self.approach_height = p('approach_height', 0.04).value
        self.lift_height = p('lift_height', 0.10).value
        self.place_drop = p('place_drop', 0.002).value
        self.retreat_distance = p('retreat_distance', 0.08).value
        self.min_fraction = p('min_cartesian_fraction', 0.98).value
        self.max_tilt = p('max_cup_tilt_deg', 5.0).value
        self.grasp_squeeze = p('grasp_squeeze', 0.002).value
        self.gripper_settle = p('gripper_settle_time', 1.0).value
        self.correct_place = p('correct_place_with_cup_pose', True).value
        self.limit_margin = p('ik_joint_limit_margin', 0.05).value
        use_sim_attach = p('use_sim_attach', False).value

        with open(cup_config) as f:
            self.cup = grasps.Cup.from_config(yaml.safe_load(f)['cup'])
        self.gripper = Gripper(self, lambda t, n: ActionClient(self, t, n, callback_group=self.cb))
        self.moveit = MoveItClient(self, self.frame, p('velocity_scaling', 0.3).value,
                                   p('acceleration_scaling', 0.3).value,
                                   p('planning_time', 10.0).value)
        self.sim_attach = None
        if use_sim_attach:  # SIMULATION-ONLY shortcut, see sim_attach.py
            from cup_task.sim_attach import SimAttach
            self.sim_attach = SimAttach(self)

        self.cup_pose = None
        self.joints = {}
        self.create_subscription(PoseStamped, self.cup_pose_topic, self._on_cup_pose, 10,
                                 callback_group=self.cb)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10,
                                 callback_group=self.cb)

    # ---- inputs --------------------------------------------------------------
    def _on_cup_pose(self, msg):
        self.cup_pose = msg

    def _on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def wait_settled(self, tolerance=0.002, window=0.3, timeout=20.0):
        """
        Wait until the arm has stopped (joint positions steady for `window` s).

        MoveIt rejects a new path whose start differs from the arm's actual
        position, and the controller reports a move done while the arm is still
        converging.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            before = [self.joints[n] for n in self.arm_joints]
            self.sim_sleep(window)
            after = [self.joints[n] for n in self.arm_joints]
            if max(abs(a - b) for a, b in zip(after, before)) < tolerance:
                return
        self.get_logger().warn('arm still moving; continuing anyway')

    def wait_cup_pose(self, timeout=30.0):
        end = time.monotonic() + timeout
        while self.cup_pose is None:
            if time.monotonic() > end:
                raise TaskError(f'no cup pose on {self.cup_pose_topic} (is a cup pose '
                                'source running?)')
            time.sleep(0.1)
        msg = self.cup_pose
        if msg.header.frame_id != self.frame:
            raise TaskError(f'cup pose is in "{msg.header.frame_id}", expected "{self.frame}"')
        o = msg.pose.orientation
        q = (o.x, o.y, o.z, o.w)
        base = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        return base, grasps.yaw_of(q), grasps.tilt_deg(q)

    def opening(self):
        return self.gripper.opening_from([self.joints[n] for n in self.gripper.joint_names])

    # ---- planning scene ------------------------------------------------------
    def cup_object(self, base, yaw):
        """
        Planning-scene cup from the cup config: body cylinder + handle box.

        The body starts 4 mm above the cup base, so a cup standing on the table
        does not count as colliding with it.
        """
        c = self.cup
        lift = 0.004
        obj = CollisionObject(id=CUP_ID)
        quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        body_h = c.height - lift
        obj.primitives.append(SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                             dimensions=[body_h, c.radius]))
        obj.primitive_poses.append(pose_msg(base + UP * (lift + body_h / 2), quat))
        reach = c.handle_radius + c.handle_thickness / 2
        handle_dir = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        centre = base + handle_dir * (c.radius + reach / 2) + UP * c.handle_center_height
        obj.primitives.append(SolidPrimitive(
            type=SolidPrimitive.BOX, dimensions=[reach, c.handle_thickness, 2 * reach]))
        obj.primitive_poses.append(pose_msg(centre, quat))
        return obj

    # ---- grasp selection -----------------------------------------------------
    def arm_seed(self, joints, opening):
        """
        IK seed/state: given joints plus the gripper at an opening.

        Drops gz_ros2_control's '<joint>_mimic' state names, which MoveIt does
        not know (it rejects robot states containing them).
        """
        seed = {n: v for n, v in joints.items() if not n.endswith('_mimic')}
        seed.update(zip(self.gripper.joint_names, self.gripper.joints_for(opening)))
        return seed

    def plan_grasp(self, base, yaw):
        """First candidate grasp whose whole pick-and-place is reachable."""
        g = self.gripper
        place_base = np.array([self.place_xy[0], self.place_xy[1], base[2]])
        current = self.arm_seed(self.joints, g.max_opening)
        held = self.cup.outer_diameter
        for grasp in grasps.candidates(self.cup, yaw, g.geometry, self.height_fraction,
                                       self.table_clearance, self.pitches):
            quat = grasps.quat_from_matrix(grasp.rotation)
            ee = grasps.ee_position(grasp, base, g.geometry)
            back = -grasp.approach * self.approach_distance
            poses = {
                'pre-grasp': (ee + back + UP * self.approach_height, g.max_opening),
                'grasp': (ee, g.max_opening),
                'lift': (ee + UP * self.lift_height, held),
                'above place': (ee - base + place_base + UP * self.lift_height, held),
                'place': (ee - base + place_base + UP * self.place_drop, held),
                'retreat': (ee - base + place_base - grasp.approach * self.retreat_distance
                            + UP * self.place_drop, g.max_opening),
            }
            solutions, seed, failed = {}, current, None
            for name in ('grasp', 'pre-grasp', 'lift', 'above place', 'place', 'retreat'):
                pos, opening = poses[name]
                sol = self.moveit.solve_ik(g.arm_group, g.ee_link, pos, quat,
                                           self.arm_seed(seed, opening), attempts=4,
                                           joints=self.arm_joints,
                                           limit_margin=self.limit_margin)
                if sol is None:
                    failed = name
                    break
                solutions[name], seed = sol, sol
            if failed:
                self.get_logger().info(f'  grasp "{grasp.name}": {failed} pose unreachable')
                continue
            self.get_logger().info(f'  grasp "{grasp.name}": all poses reachable -> using it')
            return grasp, quat, poses, solutions
        raise TaskError('no reachable grasp for this cup pose (try turning the cup or '
                        'moving it closer)')

    # ---- the task ------------------------------------------------------------
    def run(self):
        log = self.get_logger().info
        g = self.gripper
        self.moveit.wait_ready()
        # The arm's joints, taken from the SRDF group via its home pose.
        self.arm_joints = list(self.moveit.named_pose(g.arm_group, self.home_pose))
        if self.sim_attach:
            self.sim_attach.detach()  # make sure the cup starts free
        if not g.client.wait_for_server(timeout_sec=30.0):
            raise TaskError('gripper controller action not available')
        base, yaw, tilt = self.wait_cup_pose()
        margin = grasps.check_fits(self.cup, g.geometry)
        log(f'Cup at ({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}), handle yaw '
            f'{math.degrees(yaw):.0f} deg, tilt {tilt:.1f} deg. Cup '
            f'{self.cup.outer_diameter * 1000:.0f} mm, gripper opens {g.max_opening * 1000:.0f} '
            f'mm -> {margin * 1000:.1f} mm clearance per side; fingers stay '
            f'{grasps.handle_clearance(self.cup) * 1000:.0f} mm from the handle.')
        if tilt > self.max_tilt:
            raise TaskError('cup is not upright; refusing to grasp')
        self.moveit.add_object(self.cup_object(base, yaw))
        self.cup_base = base

        steps = [
            ('Lift clear of the table', self.step_lifted),
            ('Open the gripper', self.step_open),
            ('Move above the cup', self.step_pre_grasp),
            ('Descend to the grasp', lambda: self.straight('grasp')),
            ('Close the gripper', self.step_close),
            ('Lift the cup', lambda: self.straight('lift', check_level=True)),
            ('Move to the place pose', lambda: self.straight('above place', check_level=True)),
            ('Lower the cup', lambda: self.straight('place', check_level=True)),
            ('Open the gripper', self.step_release),
            ('Retreat', self.step_retreat),
            ('Return home', lambda: self.named(self.home_pose)),
        ]
        # The grasp is chosen right after step 1, with the arm above the table,
        # so IK starts from a sensible arm posture.
        for i, (title, action) in enumerate(steps, 1):
            log(f'[{i:2d}/{len(steps)}] {title}...')
            t0 = time.monotonic()
            action()
            if i == 1:
                log('       choosing a grasp (wrist turned away from the handle):')
                self.grasp, self.quat, self.poses, self.solutions = self.plan_grasp(base, yaw)
            log(f'[{i:2d}/{len(steps)}] {title}: done ({time.monotonic() - t0:.1f} s)')
        log('Pick-and-place finished.')

    # ---- steps ---------------------------------------------------------------
    def named(self, name):
        self.wait_settled()
        self.moveit.move_to_joints(self.gripper.arm_group,
                                   self.moveit.named_pose(self.gripper.arm_group, name))

    def step_lifted(self):
        self.named(self.lifted_pose)

    def step_pre_grasp(self):
        arm = {n: self.solutions['pre-grasp'][n] for n in self.arm_joints}
        self.wait_settled()
        self.moveit.move_to_joints(self.gripper.arm_group, arm)

    def straight(self, name, check_level=False):
        pos, _ = self.poses[name]
        self.wait_settled()
        self.moveit.move_straight(self.gripper.arm_group, self.gripper.ee_link,
                                  [pose_msg(pos, self.quat)], self.min_fraction)
        if check_level:
            _, _, tilt = self.wait_cup_pose()
            self.get_logger().info(f'       cup tilt {tilt:.1f} deg')
            if tilt > self.max_tilt:
                raise TaskError(f'cup tilted {tilt:.1f} deg (limit {self.max_tilt}); stopping')

    def sim_sleep(self, seconds):
        """Sleep in ROS time (simulation time in Gazebo, which runs slower)."""
        end = self.get_clock().now().nanoseconds + seconds * 1e9
        while self.get_clock().now().nanoseconds < end:
            time.sleep(0.02)

    def set_gripper(self, opening):
        """
        Command a jaw opening, wait for the move, return the measured opening.

        The controller's own result is not awaited: a gripper blocked by an
        object may never report "goal reached".
        """
        handle = wait(self.gripper.client.send_goal_async(self.gripper.goal(opening)), 30.0,
                      'gripper goal')
        if not handle.accepted:
            raise TaskError('gripper goal rejected')
        self.sim_sleep(self.gripper.close_duration + self.gripper_settle)
        return self.opening()

    def step_open(self):
        target = self.gripper.max_opening
        opening = self.set_gripper(target)
        if abs(opening - target) > 0.005:
            raise TaskError(f'gripper only opened to {opening * 1000:.0f} mm '
                            f'(wanted {target * 1000:.0f} mm); something is blocking it')

    def step_close(self):
        # 1) Close until the fingers stall on the cup.
        stalled = self.set_gripper(0.0)
        self.get_logger().info(f'       jaw stopped at {stalled * 1000:.0f} mm '
                               f'(cup is {self.cup.outer_diameter * 1000:.0f} mm)')
        if stalled < 0.6 * self.cup.outer_diameter:
            raise TaskError('gripper closed on nothing: the cup was missed')
        # (sim only) Lock the cup now: Gazebo slows to a crawl while the
        # fingers press hard into it, and the lock switches hand-cup contacts off.
        if self.sim_attach:
            self.sim_attach.attach()
        # 2) Hold just inside the measured width: a gentle, bounded squeeze
        #    instead of driving the fingers on towards fully closed.
        self.set_gripper(stalled - self.grasp_squeeze)
        self.moveit.attach(CUP_ID, self.gripper.ee_link, self.gripper.touch_links)
        if self.correct_place:
            self.correct_for_cup_in_hand()

    def correct_for_cup_in_hand(self):
        """
        Shift the place moves by how far the cup slid while the fingers closed.

        Needs a cup pose source that still sees the cup in the hand.
        """
        base, _, _ = self.wait_cup_pose()
        shift = base - self.cup_base
        self.get_logger().info(
            f'       cup moved {np.linalg.norm(shift) * 1000:.0f} mm while the fingers closed '
            f'({shift[0] * 1000:+.0f}, {shift[1] * 1000:+.0f}, {shift[2] * 1000:+.0f}) mm; '
            'place moves corrected')
        for name in ('above place', 'place', 'retreat'):
            pos, opening = self.poses[name]
            self.poses[name] = (pos - shift, opening)

    def step_release(self):
        # Open while the cup is still (sim-)locked, so the fingers leave the
        # cup before Gazebo starts checking hand-cup contacts again.
        self.step_open()
        if self.sim_attach:
            self.sim_attach.detach()
        self.moveit.detach(CUP_ID, self.gripper.ee_link)

    def step_retreat(self):
        pos, _ = self.poses['retreat']
        up = pos + UP * self.lift_height
        self.wait_settled()
        self.moveit.move_straight(self.gripper.arm_group, self.gripper.ee_link,
                                  [pose_msg(pos, self.quat), pose_msg(up, self.quat)],
                                  self.min_fraction)
        base, _, tilt = self.wait_cup_pose()
        err = math.hypot(base[0] - self.place_xy[0], base[1] - self.place_xy[1])
        self.get_logger().info(f'       cup placed at ({base[0]:.3f}, {base[1]:.3f}), '
                               f'{err * 1000:.0f} mm from the target, tilt {tilt:.1f} deg')


def main():
    rclpy.init()
    node = PickPlace()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except TaskError as e:
        node.get_logger().error(f'Pick-and-place stopped: {e}')
        raise SystemExit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
