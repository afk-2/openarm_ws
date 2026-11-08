"""
Pick-and-place of a (possibly full) cup with a body grasp, keeping it level.

Sequence, logged step by step:
  lift clear of the table -> open -> move above the cup -> descend -> close
  -> lift -> move to the place pose -> lower -> open -> retreat -> home

Inputs (all swappable):
  /cup_pose        geometry_msgs/PoseStamped from any cup pose source
                   (base centre of the cup, x axis towards the handle)
  handle_visible_topic (optional) std_msgs/Bool after each pose: whether the
                   source actually saw the handle (the camera-based estimator
                   guesses the direction of a handle it cannot see)
  cup config       cup dimensions (cup_description/config/cup.yaml by default)
  gripper config   joint names, max opening, finger size (gripper_*.yaml)

Grasp choice: the hand may approach from any direction around the cup.
Directions that would put a finger close to the handle are dropped; the
rest are ranked by how comfortably the arm reaches the grasp (how far its
joints stay from their limits) and tried in that order. Before moving,
every pose of a grasp (approach, grasp, lift, carry, place, retreat) is
checked with IK, so the robot does not start a motion it cannot finish, and
MoveIt checks that the closed fingers stay at least handle_min_clearance
from the handle. The hand orientation never changes while the cup is held, so
the cup stays level.

Pose source that cannot see the cup in the hand (pose_source_sees_held_cup:
false, e.g. the camera estimator, whose self-filter removes a held cup with
the hand): the in-hand tilt checks and the slip correction are skipped with
a warning, the grasp is confirmed from the finger width instead, and the
placement is checked on a fresh estimate taken after the hand has left.

Hidden handle (handle_visible false): its direction is only a guess. A
"handle could be here" zone (hidden_handle_spread_deg either side of the
guess) goes into the planning scene, and the clearance check covers a
handle anywhere in that zone.
"""
from collections import deque
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
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool

from cup_task import grasps
from cup_task.gripper import Gripper
from cup_task.moveit_client import MoveItClient, TaskError, pose_msg, wait

CUP_ID = 'ceramic_cup'  # same id as the world obstacle, so it gets replaced
PROBE_ID = 'handle_probe'  # temporary object for the hidden-handle clearance check
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
        self.pose_samples = max(1, p('cup_pose_samples', 5).value)
        self.sees_held_cup = p('pose_source_sees_held_cup', True).value
        self.handle_visible_topic = p('handle_visible_topic', '').value
        self.hidden_spread = p('hidden_handle_spread_deg', 45.0).value
        self.min_clearance = p('handle_min_clearance', 0.003).value
        self.visible_margin = p('visible_handle_margin_deg', 10.0).value
        self.heading_step = p('grasp_heading_step_deg', 30.0).value
        self.search_group = p('grasp_search_group', '').value
        self.finger_min_angle = p('handle_finger_min_angle_deg', 25.0).value
        self.refuse_hidden = p('refuse_hidden_handle', False).value
        self.max_stall_over = p('max_stall_over_diameter', 0.006).value
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

        self.samples = deque(maxlen=50)  # [pose message, handle visible or None]
        self.lock = threading.Lock()
        self.joints = {}
        self.warned = set()
        self.create_subscription(PoseStamped, self.cup_pose_topic, self._on_cup_pose, 10,
                                 callback_group=self.cb)
        if self.handle_visible_topic:
            self.create_subscription(Bool, self.handle_visible_topic, self._on_handle_visible,
                                     10, callback_group=self.cb)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10,
                                 callback_group=self.cb)

    # ---- inputs --------------------------------------------------------------
    def _on_cup_pose(self, msg):
        with self.lock:
            self.samples.append([msg, None])

    def _on_handle_visible(self, msg):
        # The estimator publishes this right after each pose.
        with self.lock:
            if self.samples and self.samples[-1][1] is None:
                self.samples[-1][1] = msg.data

    def warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warn(text)

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

    def wait_cup_pose(self, samples=1, newer_than=None, timeout=30.0):
        """
        Cup pose from the next `samples` estimates stamped after `newer_than`.

        Returns (base, yaw, tilt, handle_visible); handle_visible is None when
        no handle_visible_topic is used. Several samples: median position,
        average handle direction, worst tilt; if the handle was seen in some
        estimates and not in others, the majority decides.
        """
        need_visible = bool(self.handle_visible_topic)
        after = newer_than.nanoseconds if newer_than is not None else -1
        end = time.monotonic() + timeout
        while True:
            with self.lock:
                got = [(m, v) for m, v in self.samples
                       if Time.from_msg(m.header.stamp).nanoseconds > after
                       and (v is not None or not need_visible)]
            if len(got) >= samples:
                break
            if time.monotonic() > end:
                raise TaskError(f'no {"new " if newer_than else ""}cup pose on '
                                f'{self.cup_pose_topic} (is a cup pose source running, and '
                                'can it see the cup?)')
            time.sleep(0.05)
        got = got[-samples:]
        for msg, _ in got:
            if msg.header.frame_id != self.frame:
                raise TaskError(f'cup pose is in "{msg.header.frame_id}", expected '
                                f'"{self.frame}"')
        visible = None
        if need_visible:
            seen = sum(v for _, v in got)
            visible = 2 * seen > len(got)
            if 0 < seen < len(got):
                self.get_logger().info(f'       handle seen in {seen} of {len(got)} estimates; '
                                       f'going with "{"seen" if visible else "not seen"}"')
            got = [(m, v) for m, v in got if v == visible]
        quats = [(m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z,
                  m.pose.orientation.w) for m, _ in got]
        base = np.median([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                          for m, _ in got], axis=0)
        yaws = [grasps.yaw_of(q) for q in quats]
        yaw = math.atan2(sum(map(math.sin, yaws)), sum(map(math.cos, yaws)))
        return base, yaw, max(grasps.tilt_deg(q) for q in quats), visible

    def opening(self):
        return self.gripper.opening_from([self.joints[n] for n in self.gripper.joint_names])

    # ---- planning scene ------------------------------------------------------
    def handle_box(self, base, yaw, pad=0.0):
        """Box around the handle pointing at `yaw`, grown by `pad` on every side."""
        c = self.cup
        reach = c.handle_radius + c.handle_thickness / 2
        handle_dir = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        centre = base + handle_dir * (c.radius + reach / 2) + UP * c.handle_center_height
        box = SolidPrimitive(type=SolidPrimitive.BOX,
                             dimensions=[reach + 2 * pad, c.handle_thickness + 2 * pad,
                                         2 * reach + 2 * pad])
        return box, pose_msg(centre, (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))

    def cup_object(self, base, yaw, handle_spread_deg=0.0):
        """
        Planning-scene cup from the cup config: body cylinder + handle box.

        The body starts 4 mm above the cup base, so a cup standing on the table
        does not count as colliding with it. With handle_spread_deg > 0 (handle
        not seen) there is a handle box every <= 15 deg within that angle either
        side of `yaw`: the "handle could be here" zone.
        """
        c = self.cup
        lift = 0.004
        obj = CollisionObject(id=CUP_ID)
        quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        body_h = c.height - lift
        obj.primitives.append(SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                             dimensions=[body_h, c.radius]))
        obj.primitive_poses.append(pose_msg(base + UP * (lift + body_h / 2), quat))
        n = math.ceil(handle_spread_deg / 15.0)
        for offset in np.linspace(-handle_spread_deg, handle_spread_deg, 2 * n + 1):
            box, pose = self.handle_box(base, yaw + math.radians(offset))
            obj.primitives.append(box)
            obj.primitive_poses.append(pose)
        return obj

    def handle_clearance(self, base, yaw, grasp_joints, offsets_deg, max_pad=0.015):
        """
        Gap (m) between the closed fingers at the grasp and a handle turned by
        each offset from `yaw`, from MoveIt's collision shapes of the hand.

        A handle box grown by a padding is tested for contact; the largest
        padding without contact is the gap (0 = touching, max_pad = at least
        that much). The cup body is taken out of the scene meanwhile (the
        closed fingers touch it on purpose); the caller puts it back.
        """
        state = self.arm_seed(grasp_joints, self.cup.outer_diameter)

        def hits(offset, pad):
            probe = CollisionObject(id=PROBE_ID)
            box, pose = self.handle_box(base, yaw + math.radians(offset), pad)
            probe.primitives.append(box)
            probe.primitive_poses.append(pose)
            self.moveit.add_object(probe)
            pairs = self.moveit.contacts(self.gripper.arm_group, state)
            return any(PROBE_ID in pair for pair in pairs)

        gaps = {}
        self.moveit.remove_object(CUP_ID)
        try:
            for offset in offsets_deg:
                if hits(offset, 0.0):
                    gaps[offset] = 0.0
                    continue
                if not hits(offset, max_pad):
                    gaps[offset] = max_pad
                    continue
                lo, hi = 0.0, max_pad
                for _ in range(6):  # 0.25 mm resolution
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if hits(offset, mid) else (mid, hi)
                gaps[offset] = lo
        finally:
            self.moveit.remove_object(PROBE_ID)
        return gaps

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

    def comfort(self, solution):
        """How far the arm joints stay from their limits: 0 = at a limit, 1 = mid-range."""
        out = 1.0
        for name in self.arm_joints:
            lower, upper = self.moveit.limits.get(name, (-math.pi, math.pi))
            half = (upper - lower) / 2
            out = min(out, min(solution[name] - lower, upper - solution[name]) / half)
        return out

    def handle_offsets(self, handle_hidden):
        """Handle directions (deg from the estimate) the fingers must stay clear of."""
        spread = self.hidden_spread if handle_hidden else self.visible_margin
        n = max(1, math.ceil(spread / 5.0))
        return [float(a) for a in np.linspace(-spread, spread, 2 * n + 1)]

    def plan_grasp(self, base, yaw, handle_hidden=False):
        """
        Most comfortable grasp whose whole pick-and-place is reachable and
        whose closed fingers stay clear of the handle (or of the whole "handle
        could be here" zone when the handle was not seen).
        """
        g = self.gripper
        log = self.get_logger().info
        place_base = np.array([self.place_xy[0], self.place_xy[1], base[2]])
        current = self.arm_seed(self.joints, g.max_opening)
        held = self.cup.outer_diameter
        offsets = self.handle_offsets(handle_hidden)
        spread = self.hidden_spread if handle_hidden else 0.0

        # 1) Every approach direction, minus those with a finger pointing
        #    (nearly) at a possible handle direction.
        headings = np.arange(0.0, 360.0, self.heading_step)
        candidates = [
            c for c in grasps.candidates(self.cup, g.geometry, self.height_fraction,
                                         self.table_clearance, self.pitches, headings)
            if min(grasps.angle_to_closing_line(yaw + math.radians(o), c)
                   for o in offsets) >= self.finger_min_angle]

        # 2) Rank by comfort at the grasp pose. A separate planning group with
        #    a global IK solver (grasp_search_group) finds many more of them;
        #    the follow-on poses below use the arm group, starting from here.
        t0 = time.monotonic()
        ranked = []
        for grasp in candidates:
            quat = grasps.quat_from_matrix(grasp.rotation)
            ee = grasps.ee_position(grasp, base, g.geometry)
            sol = self.moveit.solve_ik(self.search_group or g.arm_group, g.ee_link, ee, quat,
                                       self.arm_seed(current, g.max_opening), attempts=2,
                                       joints=self.arm_joints, limit_margin=self.limit_margin)
            if sol is not None:
                ranked.append((self.comfort(sol), grasp, quat, ee, sol))
        ranked.sort(key=lambda r: -r[0])
        log(f'       {len(candidates)} grasps keep the fingers away from the handle'
            f'{" zone" if handle_hidden else ""}; {len(ranked)} reachable at the cup '
            f'({time.monotonic() - t0:.0f} s); trying them from most to least comfortable')

        # 3) The whole sequence, then the handle clearance with MoveIt.
        for comfort, grasp, quat, ee, grasp_sol in ranked:
            rel = math.degrees(math.remainder(yaw - math.radians(grasp.heading_deg), 2 * math.pi))
            name = f'{grasp.name} (handle {rel:+.0f} deg from the approach)'
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
            solutions, seed, failed = {'grasp': grasp_sol}, grasp_sol, None
            for pose_name in ('pre-grasp', 'lift', 'above place', 'place', 'retreat'):
                pos, opening = poses[pose_name]
                sol = self.moveit.solve_ik(g.arm_group, g.ee_link, pos, quat,
                                           self.arm_seed(seed, opening), attempts=4,
                                           joints=self.arm_joints,
                                           limit_margin=self.limit_margin)
                if sol is None:
                    failed = pose_name
                    break
                solutions[pose_name], seed = sol, sol
            if failed:
                log(f'  grasp "{name}": {failed} pose unreachable')
                continue
            blocked = self.blocked_straight_move(poses, quat, solutions)
            if blocked:
                log(f'  grasp "{name}": straight-line move "{blocked}" not possible')
                continue
            gaps = self.handle_clearance(base, yaw, grasp_sol, offsets)
            self.moveit.add_object(self.cup_object(base, yaw, spread))
            worst = min(gaps, key=gaps.get)
            if gaps[worst] < self.min_clearance:
                log(f'  grasp "{name}": closed fingers {gaps[worst] * 1000:.1f} mm from a '
                    f'handle at {worst:+.0f} deg -> too close')
                continue
            if handle_hidden:
                self.log_hidden_clearance(base, yaw, grasp_sol, gaps)
            chain = min(self.comfort(s) for s in solutions.values())
            log(f'  grasp "{name}": all poses reachable, comfort {comfort:.2f} at the grasp '
                f'({chain:.2f} worst along the way), fingers >= {gaps[worst] * 1000:.1f} mm '
                'from the handle -> using it')
            return grasp, quat, poses, solutions
        raise TaskError('no reachable grasp for this cup pose (try turning the cup or '
                        'moving it closer)')

    def blocked_straight_move(self, poses, quat, solutions):
        """
        First straight-line move of the task that cannot be followed, or None.

        The IK checks only cover the end poses; the hand must also be able to
        travel in a straight line between them (the solver may otherwise pick
        end poses in arm configurations that no straight path connects). Each
        move starts where the previous planned one ends, as when running.
        Collisions are checked for the approach only: during the carry the
        cup is not attached yet in this check, and the real moves are
        collision-checked again when they run.
        """
        g = self.gripper
        up = UP * self.lift_height
        moves = [('descend to the grasp', ['grasp'], True, g.max_opening),
                 ('lift', ['lift'], False, self.cup.outer_diameter),
                 ('carry', ['above place'], False, self.cup.outer_diameter),
                 ('lower', ['place'], False, self.cup.outer_diameter),
                 ('retreat', ['retreat', 'retreat up'], False, g.max_opening)]
        targets = dict(poses)
        targets['retreat up'] = (poses['retreat'][0] + up, g.max_opening)
        state = self.arm_seed(solutions['pre-grasp'], g.max_opening)
        for title, names, avoid, opening in moves:
            state = self.arm_seed(state, opening)
            fraction, state = self.moveit.straight_path(
                g.arm_group, g.ee_link, state,
                [pose_msg(targets[n][0], quat) for n in names], avoid)
            if fraction < self.min_fraction:
                return f'{title}: {fraction * 100:.0f}%'
        return None

    def log_hidden_clearance(self, base, yaw, grasp_joints, zone_gaps):
        """Log the finger gap to a hidden handle out to 90 deg either side."""
        offsets = [float(a) for a in range(-90, 91, 15)]
        gaps = self.handle_clearance(base, yaw, grasp_joints, offsets)
        self.moveit.add_object(self.cup_object(base, yaw, self.hidden_spread))
        worst = min(zone_gaps, key=zone_gaps.get)
        touch = [abs(a) for a, gap in gaps.items() if gap == 0.0]
        self.get_logger().info(
            '       closed fingers vs. hidden handle, gap (mm) by handle offset (deg): '
            + ' '.join(f'{a:+.0f}:{gap * 1000:.1f}' for a, gap in gaps.items()))
        self.get_logger().info(
            f'       zone +-{self.hidden_spread:.0f} deg: smallest gap '
            f'{zone_gaps[worst] * 1000:.1f} mm (handle at {worst:+.0f} deg); fingers would '
            'touch the handle at ' + (f'{min(touch):.0f} deg or more off' if touch
                                      else 'no offset up to 90 deg'))

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
        if not self.sees_held_cup:
            self.get_logger().warn(
                'Cup pose source cannot see the cup while it is held: the in-hand tilt '
                'checks and the slip correction are SKIPPED; the grasp is confirmed from '
                'the finger width, and the placement from a new estimate after release.')
        base, yaw, tilt, visible = self.wait_cup_pose(self.pose_samples,
                                                      newer_than=self.get_clock().now())
        hidden = visible is False
        if hidden:
            self.get_logger().warn(
                f'Handle NOT seen: relying on the assumption that it points away from the '
                f'camera (yaw {math.degrees(yaw):.0f} deg). Treating +-{self.hidden_spread:.0f} '
                'deg around that as "handle could be here".')
            if self.refuse_hidden:
                raise TaskError('handle not seen and refuse_hidden_handle is set; turn the '
                                'cup so the camera can see its handle')
        margin = grasps.check_fits(self.cup, g.geometry)
        log(f'Cup at ({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}), handle yaw '
            f'{math.degrees(yaw):.0f} deg, tilt {tilt:.1f} deg. Cup '
            f'{self.cup.outer_diameter * 1000:.0f} mm, gripper opens {g.max_opening * 1000:.0f} '
            f'mm -> {margin * 1000:.1f} mm clearance per side.')
        if tilt > self.max_tilt:
            raise TaskError('cup is not upright; refusing to grasp')
        self.moveit.add_object(self.cup_object(base, yaw, self.hidden_spread if hidden else 0.0))
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
                log('       choosing a grasp (all directions, most comfortable first):')
                self.grasp, self.quat, self.poses, self.solutions = self.plan_grasp(base, yaw,
                                                                                    hidden)
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
        if check_level and not self.sees_held_cup:
            self.warn_once('tilt', '       cup tilt NOT checked: the pose source cannot see '
                                   'the cup in the hand')
        elif check_level:
            _, _, tilt, _ = self.wait_cup_pose(newer_than=self.get_clock().now())
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
        if stalled > self.cup.outer_diameter + self.max_stall_over:
            raise TaskError(f'fingers stopped at {stalled * 1000:.0f} mm, wider than the cup: '
                            'something else (the handle?) is between them')
        # (sim only) Lock the cup now: Gazebo slows to a crawl while the
        # fingers press hard into it, and the lock switches hand-cup contacts off.
        if self.sim_attach:
            self.sim_attach.attach()
        # 2) Hold just inside the measured width: a gentle, bounded squeeze
        #    instead of driving the fingers on towards fully closed.
        self.set_gripper(stalled - self.grasp_squeeze)
        self.moveit.attach(CUP_ID, self.gripper.ee_link, self.gripper.touch_links)
        if self.correct_place and self.sees_held_cup:
            self.correct_for_cup_in_hand()
        elif self.correct_place:
            self.get_logger().warn('       slip correction SKIPPED: the pose source cannot see '
                                   'the cup in the hand')

    def correct_for_cup_in_hand(self):
        """
        Shift the place moves by how far the cup slid while the fingers closed.

        Needs a cup pose source that still sees the cup in the hand.
        """
        base, _, _, _ = self.wait_cup_pose(newer_than=self.get_clock().now())
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
        # A new estimate, taken after the hand has left (not an old message).
        try:
            base, _, tilt, _ = self.wait_cup_pose(min(3, self.pose_samples),
                                                  newer_than=self.get_clock().now(),
                                                  timeout=20.0)
        except TaskError as e:
            self.get_logger().warn(f'       placement not confirmed: {e}')
            return
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
