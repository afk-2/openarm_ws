"""
Small MoveIt client built on move_group's standard ROS interfaces.

Humble has no MoveIt Python API in its binaries, so this uses:

  /move_action               MoveGroup action: plan + execute to a joint goal
  /compute_ik                GetPositionIK: joint angles for a hand pose
  /compute_cartesian_path    GetCartesianPath: straight-line hand motion
  /execute_trajectory        ExecuteTrajectory action: run a computed path
  /apply_planning_scene      ApplyPlanningScene: add/attach/detach objects
  /check_state_validity      GetStateValidity: is a robot state in collision?
  /robot_description_semantic  SRDF, read for the named poses ("home", ...)
"""
import time
import xml.etree.ElementTree as ET

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                             JointConstraint, MoveItErrorCodes, PlanningScene, RobotState)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetPositionIK,
                             GetStateValidity)
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items()
               if k.isupper() and isinstance(v, int)}


class TaskError(RuntimeError):
    pass


def wait(future, timeout, what):
    end = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > end:
            raise TaskError(f'timed out waiting for {what}')
        time.sleep(0.02)
    return future.result()


def pose_msg(position, quat_xyzw):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat_xyzw
    return p


class MoveItClient:

    def __init__(self, node, frame, velocity_scaling, acceleration_scaling, planning_time):
        self.node = node
        self.frame = frame
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling
        self.planning_time = planning_time
        self.move_group = ActionClient(node, MoveGroup, '/move_action')
        self.execute = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        self.ik = node.create_client(GetPositionIK, '/compute_ik')
        self.cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.scene = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.validity = node.create_client(GetStateValidity, '/check_state_validity')
        self.srdf = None
        self.limits = {}
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description_semantic',
                                 lambda m: setattr(self, 'srdf', m.data), latched)
        node.create_subscription(String, '/robot_description', self._on_urdf, latched)

    def _on_urdf(self, msg):
        for joint in ET.fromstring(msg.data).iter('joint'):
            limit = joint.find('limit')
            if joint.get('type') == 'revolute' and limit is not None:
                self.limits[joint.get('name')] = (float(limit.get('lower')),
                                                  float(limit.get('upper')))

    def wait_ready(self, timeout=120.0):
        end = time.monotonic() + timeout
        for what, ok in (('move_group', lambda: self.move_group.wait_for_server(timeout_sec=1.0)),
                         ('/compute_ik', lambda: self.ik.wait_for_service(timeout_sec=1.0)),
                         ('SRDF', lambda: self.srdf is not None or time.sleep(1.0)),
                         ('URDF', lambda: bool(self.limits) or time.sleep(1.0))):
            while not ok():
                if time.monotonic() > end:
                    raise TaskError(f'MoveIt not available ({what}); is move_group running?')

    # ---- named poses -------------------------------------------------------
    def named_pose(self, group, name):
        root = ET.fromstring(self.srdf)
        for state in root.iter('group_state'):
            if state.get('group') == group and state.get('name') == name:
                return {j.get('name'): float(j.get('value')) for j in state.iter('joint')}
        raise TaskError(f'named pose "{name}" for group "{group}" not found in the SRDF')

    # ---- motion ------------------------------------------------------------
    def move_to_joints(self, group, joints, timeout=300.0):
        """Plan a collision-free path to the joint angles and execute it."""
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = group
        r.num_planning_attempts = 5
        r.allowed_planning_time = self.planning_time
        r.max_velocity_scaling_factor = self.velocity_scaling
        r.max_acceleration_scaling_factor = self.acceleration_scaling
        r.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=n, position=float(v), tolerance_above=0.005,
                            tolerance_below=0.005, weight=1.0) for n, v in joints.items()])]
        r.start_state.is_diff = True
        goal.planning_options.plan_only = False
        # A sampling planner's smoothed path occasionally clips an obstacle and
        # is rejected as invalid; plan again instead of giving up.
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        result = self.call_action(self.move_group, goal, timeout, 'planning + execution')
        self._check(result.error_code, 'plan/execute')

    def move_straight(self, group, link, waypoints, min_fraction, timeout=300.0):
        """Move the hand along straight lines through waypoints (orientation as given)."""
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.start_state.is_diff = True
        req.group_name = group
        req.link_name = link
        req.waypoints = waypoints
        req.max_step = 0.005
        req.jump_threshold = 5.0  # reject sudden joint flips
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling
        res = wait(self.cartesian.call_async(req), 60.0, 'Cartesian path')
        if res.fraction < min_fraction:
            raise TaskError(f'only {res.fraction * 100:.0f}% of the straight-line move is '
                            'possible (unreachable or in collision)')
        goal = ExecuteTrajectory.Goal(trajectory=res.solution)
        self._check(self.call_action(self.execute, goal, timeout, 'execution').error_code,
                    'execute')

    def solve_ik(self, group, link, position, quat_xyzw, seed_joints=None, attempts=5,
                 joints=(), limit_margin=0.0):
        """
        Collision-free joint angles for a hand pose, or None.

        Solutions with any of `joints` closer than limit_margin (rad) to a joint
        limit are rejected: a joint pressed against its end stop cannot follow
        the next planned path.
        """
        req = GetPositionIK.Request()
        q = req.ik_request
        q.group_name = group
        q.ik_link_name = link
        q.avoid_collisions = True
        q.timeout.nanosec = 200_000_000
        q.robot_state.is_diff = True
        if seed_joints:
            q.robot_state.joint_state = JointState(name=list(seed_joints),
                                                   position=list(seed_joints.values()))
        q.pose_stamped = PoseStamped(pose=pose_msg(position, quat_xyzw))
        q.pose_stamped.header.frame_id = self.frame
        for _ in range(attempts):
            res = wait(self.ik.call_async(req), 10.0, 'IK')
            if res.error_code.val == MoveItErrorCodes.SUCCESS:
                js = res.solution.joint_state
                solution = dict(zip(js.name, js.position))
                if self.within_limits(solution, joints, limit_margin):
                    return solution
        return None

    def within_limits(self, solution, joints, margin):
        for name in joints:
            lower, upper = self.limits.get(name, (-1e9, 1e9))
            if not lower + margin <= solution[name] <= upper - margin:
                return False
        return True

    # ---- planning scene ----------------------------------------------------
    def contacts(self, group, joints):
        """Names of the bodies in collision (pairs) for a robot state given as joints."""
        req = GetStateValidity.Request(group_name=group)
        req.robot_state.is_diff = True
        req.robot_state.joint_state = JointState(name=list(joints),
                                                 position=[float(v) for v in joints.values()])
        res = wait(self.validity.call_async(req), 10.0, 'state validity check')
        return [(c.contact_body_1, c.contact_body_2) for c in res.contacts]

    def remove_object(self, object_id):
        scene = PlanningScene(is_diff=True)
        obj = CollisionObject(id=object_id, operation=CollisionObject.REMOVE)
        obj.header.frame_id = self.frame
        scene.world.collision_objects = [obj]
        self._apply(scene, f'remove {object_id}')

    def add_object(self, obj):
        scene = PlanningScene(is_diff=True)
        obj.header.frame_id = self.frame
        obj.operation = CollisionObject.ADD
        scene.world.collision_objects = [obj]
        self._apply(scene, f'add {obj.id}')

    def attach(self, object_id, link, touch_links):
        aco = AttachedCollisionObject(link_name=link, touch_links=touch_links)
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.robot_state = RobotState(is_diff=True, attached_collision_objects=[aco])
        self._apply(scene, f'attach {object_id}')

    def detach(self, object_id, link):
        aco = AttachedCollisionObject(link_name=link)
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.robot_state = RobotState(is_diff=True, attached_collision_objects=[aco])
        self._apply(scene, f'detach {object_id}')

    # ---- helpers -----------------------------------------------------------
    def _apply(self, scene, what):
        res = wait(self.scene.call_async(ApplyPlanningScene.Request(scene=scene)), 10.0, what)
        if not res.success:
            raise TaskError(f'planning scene update failed: {what}')

    def call_action(self, client, goal, timeout, what):
        handle = wait(client.send_goal_async(goal), 30.0, f'{what} (goal)')
        if not handle.accepted:
            raise TaskError(f'{what} goal rejected')
        return wait(handle.get_result_async(), timeout, what).result

    @staticmethod
    def _check(error_code, what):
        if error_code.val != MoveItErrorCodes.SUCCESS:
            raise TaskError(f'{what} failed: {ERROR_NAMES.get(error_code.val, error_code.val)}')
