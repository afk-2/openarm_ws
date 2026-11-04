#!/usr/bin/env python3
"""
Add static obstacles from the Gazebo world to MoveIt's planning scene.

Reads the world SDF (cup_description/worlds/cup_table.sdf by default) and
turns the box/cylinder <collision> shapes of the named models into
CollisionObjects, so the planning scene always matches what Gazebo simulates.
Poses are converted from the Gazebo world frame to the robot's "world" frame
(the base of the body), i.e. shifted by the robot's spawn position.

Parameters:
  world_file  path to the world SDF
  models      model names to add (default: table, robot_pedestal, ceramic_cup)
  spawn_xyz   robot spawn position in the Gazebo world (default: [0, 0, 0.40])
  frame_id    planning frame (default: world)
  touch_links "object:link" pairs that rest on each other and may touch
              (default: the robot body standing on its pedestal)

Models pulled in with <include><uri>model://NAME</uri> are resolved against
cup_description/models.
"""
import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AllowedCollisionEntry, CollisionObject, PlanningScene,
                             PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import numpy as np
import rclpy
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


def pose_matrix(text):
    """SDF '<pose>x y z roll pitch yaw</pose>' -> 4x4 transform."""
    x, y, z, roll, pitch, yaw = [float(v) for v in (text or '0 0 0 0 0 0').split()]
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    t = np.eye(4)
    t[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    t[:3, 3] = [x, y, z]
    return t


def matrix_to_pose(t):
    m = t[:3, :3]

    def component(sx, sy, sz):
        return math.sqrt(max(0.0, 1.0 + sx * m[0, 0] + sy * m[1, 1] + sz * m[2, 2])) / 2.0

    w = component(1, 1, 1)
    x = math.copysign(component(1, -1, -1), m[2, 1] - m[1, 2])
    y = math.copysign(component(-1, 1, -1), m[0, 2] - m[2, 0])
    z = math.copysign(component(-1, -1, 1), m[1, 0] - m[0, 1])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = t[:3, 3]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w
    return pose


def primitive(geometry):
    box = geometry.find('box/size')
    if box is not None:
        return SolidPrimitive(type=SolidPrimitive.BOX,
                              dimensions=[float(v) for v in box.text.split()])
    cylinder = geometry.find('cylinder')
    if cylinder is not None:
        return SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[
            float(cylinder.find('length').text), float(cylinder.find('radius').text)])
    return None


def world_models(world_file):
    """Yield (name, model element, model pose in the Gazebo world)."""
    models_dir = os.path.join(get_package_share_directory('cup_description'), 'models')
    world = ET.parse(world_file).getroot().find('world')
    for model in world.findall('model'):
        yield model.get('name'), model, pose_matrix(model.findtext('pose'))
    for include in world.findall('include'):
        uri = include.findtext('uri')
        model = ET.parse(os.path.join(models_dir, uri.replace('model://', ''),
                                      'model.sdf')).getroot().find('model')
        yield (include.findtext('name') or model.get('name'), model,
               pose_matrix(include.findtext('pose')))


class AddWorldObstacles(Node):

    def __init__(self):
        super().__init__('add_world_obstacles')
        default_world = os.path.join(
            get_package_share_directory('cup_description'), 'worlds', 'cup_table.sdf')
        self.world_file = self.declare_parameter('world_file', default_world).value
        self.models = self.declare_parameter(
            'models', ['table', 'robot_pedestal', 'ceramic_cup']).value
        self.spawn_xyz = self.declare_parameter('spawn_xyz', [0.0, 0.0, 0.40]).value
        self.frame_id = self.declare_parameter('frame_id', 'world').value
        self.touch_links = self.declare_parameter(
            'touch_links', ['robot_pedestal:openarm_body_link0']).value
        self.apply_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.get_client = self.create_client(GetPlanningScene, '/get_planning_scene')

    def collision_objects(self):
        gz_to_robot = np.eye(4)
        gz_to_robot[:3, 3] = -np.asarray(self.spawn_xyz, dtype=float)
        objects = []
        for name, model, model_pose in world_models(self.world_file):
            if name not in self.models:
                continue
            obj = CollisionObject(id=name, operation=CollisionObject.ADD)
            obj.header.frame_id = self.frame_id
            for link in model.findall('link'):
                link_pose = model_pose @ pose_matrix(link.findtext('pose'))
                for collision in link.findall('collision'):
                    shape = primitive(collision.find('geometry'))
                    if shape is None:
                        continue  # planes and meshes are not needed here
                    obj.primitives.append(shape)
                    obj.primitive_poses.append(matrix_to_pose(
                        gz_to_robot @ link_pose @ pose_matrix(collision.findtext('pose'))))
            objects.append(obj)
            self.get_logger().info(f'{name}: {len(obj.primitives)} shapes')
        missing = set(self.models) - {o.id for o in objects}
        if missing:
            self.get_logger().warn(f'Models not found in {self.world_file}: {sorted(missing)}')
        return objects

    def call(self, client, request):
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(f'Waiting for move_group ({client.srv_name})...')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def allowed_collision_matrix(self, object_ids):
        """Current ACM plus the touch_links pairs.

        An ACM in a planning scene diff replaces the whole matrix (incl. the
        SRDF entries), so start from move_group's current one.
        """
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm = self.call(self.get_client, request).scene.allowed_collision_matrix
        for pair in self.touch_links:
            obj, link = pair.split(':')
            if obj not in object_ids:
                continue
            for name in (obj, link):
                if name not in acm.entry_names:
                    acm.entry_names.append(name)
                    for entry in acm.entry_values:
                        entry.enabled.append(False)
                    acm.entry_values.append(
                        AllowedCollisionEntry(enabled=[False] * len(acm.entry_names)))
            i, j = acm.entry_names.index(obj), acm.entry_names.index(link)
            acm.entry_values[i].enabled[j] = acm.entry_values[j].enabled[i] = True
        return acm

    def run(self):
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = self.collision_objects()
        scene.allowed_collision_matrix = self.allowed_collision_matrix(
            {o.id for o in scene.world.collision_objects})
        result = self.call(self.apply_client, ApplyPlanningScene.Request(scene=scene))
        if result is not None and result.success:
            self.get_logger().info('Planning scene updated')
            return True
        self.get_logger().error('Failed to apply planning scene')
        return False


def main():
    rclpy.init()
    ok = AddWorldObstacles().run()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
