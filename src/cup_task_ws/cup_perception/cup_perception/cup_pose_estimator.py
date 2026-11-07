"""
Cup poses from the D435 depth image.

Pipeline (thresholds in config/cup_pose_estimator.yaml):
  1. depth image -> 3D points (camera frame) -> world frame via TF
  2. crop to a box above the table
  3. find the table plane (RANSAC) and keep the points above it
  4. remove points on the robot itself (URDF collision shapes placed via TF)
  5. cluster (DBSCAN) and keep every cluster with a cup-like size
  6. per cup: circle fit on the outer wall -> centre; table plane -> base
     height; points outside the wall -> handle direction (or, if the handle is
     not visible, assume it points away from the camera)

Outputs (world frame). Pose convention as /cup_pose: centre of the cup's base,
x axis towards the handle.
  /cup_pose_estimate                 PoseStamped, nearest reachable cup
  /cup_pose_estimate/handle_visible  Bool, whether its handle was seen
  /cup_poses_estimate                PoseArray, every cup found
  /cup_perception/markers            MarkerArray: cup cylinders, handle arrows, labels
  /cup_perception/status             String, one summary line per frame
Debug clouds (PointCloud2): /cup_perception/{cropped, above_table,
  robot_points, objects, cup_clusters, outer_wall, handle_points}
"""
from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, ColorRGBA, Header, String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from cup_perception import cloud_ops
from cup_perception.robot_filter import RobotSelfFilter

DEBUG_CLOUDS = ('cropped', 'above_table', 'robot_points', 'objects', 'cup_clusters',
                'outer_wall', 'handle_points')


def matrix_from_transform(t):
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [p.x, p.y, p.z]
    return m


@dataclass
class CupEstimate:
    center: np.ndarray    # x, y
    base_z: float
    radius: float
    height: float
    yaw: float            # handle direction
    handle_visible: bool
    wall_points: np.ndarray
    handle_points: np.ndarray
    arc_deg: float

    def pose(self):
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(self.center[0]),
                                                    float(self.center[1]), self.base_z)
        p.orientation.z, p.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        return p


class CupPoseEstimator(Node):

    def __init__(self):
        super().__init__('cup_pose_estimator')

        def p(name, default):
            return self.declare_parameter(name, default).value

        self.world = p('world_frame', 'world')
        self.period = 1.0 / p('max_rate', 2.0)
        self.stride = p('pixel_stride', 2)
        self.range = (p('min_range', 0.2), p('max_range', 1.5))
        self.crop = (p('crop_min', [0.10, -0.60, 0.30]), p('crop_max', [0.95, 0.60, 0.60]))
        self.plane_cfg = (p('plane_distance', 0.006), p('plane_max_tilt_deg', 10.0),
                          p('plane_iterations', 150))
        self.min_above = p('min_height_above_table', 0.008)
        self.self_filter_cfg = (p('self_filter_enabled', True), p('self_filter_spacing', 0.005),
                                p('self_filter_margin', 0.015))
        self.cluster_cfg = (p('cluster_eps', 0.015), p('cluster_min_samples', 8))
        self.cup_cfg = (p('cup_min_points', 40), p('cup_height_range', [0.03, 0.12]),
                        p('cup_footprint_range', [0.03, 0.12]))
        self.wall_band = p('wall_height_band', [0.2, 0.85])
        fixed = p('cup_outer_diameter', 0.065) / 2 if p('circle_fix_radius', True) else 0.0
        self.circle_cfg = (p('circle_radius_range', [0.02, 0.05]), p('circle_tolerance', 0.002),
                           p('circle_iterations', 300), fixed)
        self.min_wall_points = p('circle_min_inliers', 30)
        self.min_arc = p('circle_min_arc_deg', 60.0)
        self.handle_cfg = (p('handle_gap', 0.004), p('handle_max_reach', 0.035),
                           p('handle_height_band', [0.1, 0.95]), p('handle_min_points', 15),
                           p('handle_cluster_eps', 0.006), p('handle_cluster_min_samples', 10),
                           p('handle_min_reach', 0.010))
        self.reach_origin = np.array(p('reach_origin_xy', [0.0, -0.1525]))
        self.reach_max = p('reach_max_distance', 0.45)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_info = None
        self.self_filter = None
        self.last_run = 0.0
        self.last_depth = None
        self.pub = {name: self.create_publisher(point_cloud2.PointCloud2,
                                                f'/cup_perception/{name}', 2)
                    for name in DEBUG_CLOUDS}
        self.pose_pub = self.create_publisher(PoseStamped, '/cup_pose_estimate', 10)
        self.visible_pub = self.create_publisher(Bool, '/cup_pose_estimate/handle_visible', 10)
        self.array_pub = self.create_publisher(PoseArray, '/cup_poses_estimate', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cup_perception/markers', 10)
        self.status_pub = self.create_publisher(String, '/cup_perception/status', 10)

        info_topic = p('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
        self.depth_topic = p('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.create_subscription(Image, self.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(String, '/robot_description', self.on_urdf,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Wall-clock timer, so it also fires when no simulation clock is running.
        self.create_timer(5.0, self.check_input, clock=rclpy.clock.Clock())

    # ---- inputs ----------------------------------------------------------------
    def check_input(self):
        if self.last_depth is None or time.monotonic() - self.last_depth > 5.0:
            self.get_logger().warn(
                f'no depth images on {self.depth_topic} for 5 s: is the simulation '
                '(or the camera driver) running?')

    def on_info(self, msg):
        self.camera_info = msg

    def on_urdf(self, msg):
        enabled, spacing, margin = self.self_filter_cfg
        if enabled:
            self.self_filter = RobotSelfFilter(msg.data, spacing, margin, self.get_logger())
            n = sum(len(s) for s in self.self_filter.samples.values())
            self.get_logger().info(f'self-filter: {len(self.self_filter.samples)} links, '
                                   f'{n} sample points, margin {margin * 1000:.0f} mm')

    def on_depth(self, msg):
        self.last_depth = time.monotonic()
        if self.camera_info is None or time.monotonic() - self.last_run < self.period:
            return
        self.last_run = time.monotonic()
        try:
            self.process(msg)
        except (ValueError, tf2_ros.TransformException) as e:
            self.get_logger().warn(f'skipping frame: {e}', throttle_duration_sec=5.0)

    # ---- helpers ---------------------------------------------------------------
    def lookup(self, frame, stamp=None, timeout=0.2):
        """World <- frame transform at stamp (None: the latest available)."""
        stamp = stamp if stamp is not None else rclpy.time.Time()
        return matrix_from_transform(self.tf_buffer.lookup_transform(
            self.world, frame, stamp, timeout=Duration(seconds=timeout)).transform)

    def publish_cloud(self, name, points, stamp):
        header = Header(stamp=stamp, frame_id=self.world)
        self.pub[name].publish(point_cloud2.create_cloud_xyz32(
            header, np.asarray(points, dtype=np.float32).reshape(-1, 3)))

    # ---- pipeline --------------------------------------------------------------
    def process(self, msg):
        t0 = time.monotonic()
        stamp = msg.header.stamp
        # 1. Depth -> points in the camera frame -> world frame.
        depth = cloud_ops.depth_to_metres(msg.data, msg.encoding, msg.height, msg.width, msg.step)
        k = self.camera_info.k
        pts_cam = cloud_ops.depth_to_points(depth, k[0], k[4], k[2], k[5], self.stride,
                                            *self.range)
        cam_tf = self.lookup(msg.header.frame_id, stamp)
        pts = cloud_ops.transform(pts_cam, cam_tf)
        camera_xy = cam_tf[:2, 3]

        # 2. Crop to the box above the table.
        cropped = cloud_ops.crop_box(pts, *self.crop)
        self.publish_cloud('cropped', cropped, stamp)

        # 3. Table plane and the points above it.
        plane = cloud_ops.fit_table_plane(cropped, *self.plane_cfg)
        if plane is None:
            self.get_logger().warn('no table plane found', throttle_duration_sec=5.0)
            return
        above = cropped[plane.height_of(cropped) > self.min_above]
        self.publish_cloud('above_table', above, stamp)

        # 4. Remove points on the robot.
        on_robot = np.zeros((0, 3))
        objects = above
        if self.self_filter is not None:
            # Latest link poses, not the image's exact stamp: waiting for each of
            # ~26 links would stall the frame, and the margin absorbs the
            # small time difference.
            links = {}
            for name in self.self_filter.samples:
                try:
                    links[name] = self.lookup(name, timeout=0.0)
                except tf2_ros.TransformException:
                    pass
            robot = self.self_filter.robot_points(links, *self.crop)
            objects, on_robot = self.self_filter.split(above, robot)
        self.publish_cloud('robot_points', on_robot, stamp)
        self.publish_cloud('objects', objects, stamp)

        # 5. Clusters with a cup-like size.
        labels = cloud_ops.dbscan(objects, *self.cluster_cfg)
        clusters = cloud_ops.describe_clusters(objects, labels, plane)
        candidates, rejected = cloud_ops.select_cups(clusters, *self.cup_cfg)
        self.publish_cloud('cup_clusters', np.vstack([c.points for c in candidates])
                           if candidates else [], stamp)

        # 6. Pose of each cup.
        cups, notes = [], []
        for i, c in enumerate(candidates):
            cup, note = self.estimate(c, plane, camera_xy)
            if cup is None:
                notes.append(f'cluster {i}: {note}')
            else:
                cups.append(cup)
        self.publish_cloud('outer_wall', np.vstack([c.wall_points for c in cups])
                           if cups else [], stamp)
        handle = [c.handle_points for c in cups if len(c.handle_points)]
        self.publish_cloud('handle_points', np.vstack(handle) if handle else [], stamp)
        target = self.pick_target(cups)
        self.publish_results(cups, target, stamp)

        status = (f'{len(pts)} pts, table z {plane.z_at(0.5, 0.0):.3f}, above {len(above)}, '
                  f'on robot {len(on_robot)}, {len(clusters)} clusters, '
                  f'{len(candidates)} cup-sized, {len(cups)} cups')
        for i, cup in enumerate(cups):
            status += (f' | cup {i}{" (target)" if cup is target else ""}: '
                       f'({cup.center[0]:.3f}, {cup.center[1]:.3f}, {cup.base_z:.3f}) '
                       f'r {cup.radius * 1000:.1f} mm, arc {cup.arc_deg:.0f} deg, handle '
                       + (f'{math.degrees(cup.yaw):.0f} deg' if cup.handle_visible
                          else f'not visible (assumed {math.degrees(cup.yaw):.0f} deg)'))
        if notes:
            status += ' | rejected: ' + '; '.join(notes[:3])
        # Clusters big enough to matter that failed the size check (small
        # noise clusters are left out).
        big = [r for r in rejected if 'points (<' not in r]
        if not candidates and rejected:
            status += f' | no cup-sized cluster ({"; ".join((big or rejected)[:3])})'
        elif big:
            status += ' | size rejected: ' + '; '.join(big[:3])
        status += f' [{(time.monotonic() - t0) * 1000:.0f} ms]'
        self.status_pub.publish(String(data=status))
        self.get_logger().info(status, throttle_duration_sec=2.0)

    def estimate(self, cluster, plane, camera_xy):
        pts = cluster.points
        heights = plane.height_of(pts)
        band = ((heights >= self.wall_band[0] * cluster.height)
                & (heights <= self.wall_band[1] * cluster.height))
        wall = pts[band]
        circle = cloud_ops.fit_outer_circle(wall[:, :2], camera_xy, *self.circle_cfg)
        if circle is None:
            return None, 'no cup-like circle'
        if circle.inliers.sum() < self.min_wall_points:
            return None, f'only {int(circle.inliers.sum())} wall points'
        if circle.arc_deg < self.min_arc:
            return None, f'wall arc only {circle.arc_deg:.0f} deg'
        yaw, handle_pts = cloud_ops.estimate_handle(pts, heights, circle, cluster.height,
                                                    *self.handle_cfg)
        visible = yaw is not None
        if not visible:  # assume the handle points away from the camera
            away = circle.center - camera_xy
            yaw = math.atan2(away[1], away[0])
        base_z = plane.z_at(*circle.center)
        return CupEstimate(circle.center, base_z, circle.radius, cluster.height, yaw, visible,
                           wall[circle.inliers], handle_pts, circle.arc_deg), ''

    def pick_target(self, cups):
        """Nearest cup within reach_max_distance of reach_origin_xy (or None)."""
        best, best_d = None, self.reach_max
        for cup in cups:
            d = float(np.linalg.norm(cup.center - self.reach_origin))
            if d <= best_d:
                best, best_d = cup, d
        return best

    # ---- outputs ---------------------------------------------------------------
    def publish_results(self, cups, target, stamp):
        header = Header(stamp=stamp, frame_id=self.world)
        self.array_pub.publish(PoseArray(header=header, poses=[c.pose() for c in cups]))
        if target is not None:
            self.pose_pub.publish(PoseStamped(header=header, pose=target.pose()))
            self.visible_pub.publish(Bool(data=target.handle_visible))
        markers = [Marker(header=header, action=Marker.DELETEALL)]
        for i, cup in enumerate(cups):
            chosen = cup is target
            body = Marker(header=header, ns='cup', id=i, type=Marker.CYLINDER)
            body.pose = cup.pose()
            body.pose.position.z += cup.height / 2
            body.scale.x = body.scale.y = 2 * cup.radius
            body.scale.z = cup.height
            body.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.6 if chosen else 0.3)
            arrow = Marker(header=header, ns='handle', id=i, type=Marker.ARROW)
            z = cup.base_z + 0.6 * cup.height
            start = Point(x=float(cup.center[0]), y=float(cup.center[1]), z=z)
            end = Point(x=start.x + 0.08 * math.cos(cup.yaw),
                        y=start.y + 0.08 * math.sin(cup.yaw), z=z)
            arrow.points = [start, end]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.006, 0.012, 0.015
            arrow.color = (ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0) if cup.handle_visible
                           else ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8))
            label = Marker(header=header, ns='label', id=i, type=Marker.TEXT_VIEW_FACING)
            label.pose = cup.pose()
            label.pose.position.z += cup.height + 0.04
            label.scale.z = 0.018
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = (f'cup {i}{" - TARGET" if chosen else ""}\nhandle '
                          + ('seen' if cup.handle_visible else 'not seen (assumed)'))
            markers += [body, arrow, label]
        self.marker_pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    rclpy.spin(CupPoseEstimator())


if __name__ == '__main__':
    main()
