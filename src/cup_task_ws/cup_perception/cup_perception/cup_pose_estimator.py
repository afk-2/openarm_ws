"""
Cup pose from the D435 depth image.

Pipeline (thresholds in config/cup_pose_estimator.yaml):
  1. depth image -> 3D points (camera frame) -> world frame via TF
  2. crop to a box above the table
  3. find the table plane (RANSAC) and keep the points above it
  4. cluster (DBSCAN) and pick the cluster with a cup-like size

Debug outputs (sensor_msgs/PointCloud2, world frame), for RViz:
  /cup_perception/cropped        step 2
  /cup_perception/above_table    step 3
  /cup_perception/cup_cluster    step 4
"""
import math
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import tf2_ros

from cup_perception import cloud_ops


def matrix_from_transform(t):
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [p.x, p.y, p.z]
    return m


class CupPoseEstimator(Node):

    def __init__(self):
        super().__init__('cup_pose_estimator')
        p = self.declare_parameter
        self.world = p('world_frame', 'world').value
        self.period = 1.0 / p('max_rate', 2.0).value
        self.stride = p('pixel_stride', 2).value
        self.range = (p('min_range', 0.2).value, p('max_range', 1.5).value)
        self.crop = (p('crop_min', [0.10, -0.60, 0.30]).value,
                     p('crop_max', [0.95, 0.60, 0.60]).value)
        self.plane_cfg = (p('plane_distance', 0.006).value, p('plane_max_tilt_deg', 10.0).value,
                          p('plane_iterations', 150).value)
        self.min_above = p('min_height_above_table', 0.008).value
        self.cluster_cfg = (p('cluster_eps', 0.015).value, p('cluster_min_samples', 8).value)
        self.cup_cfg = (p('cup_min_points', 40).value, p('cup_height_range', [0.03, 0.12]).value,
                        p('cup_footprint_range', [0.03, 0.12]).value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_info = None
        self.last_run = 0.0
        self.last_depth = None
        self.pubs = {name: self.create_publisher(
            point_cloud2.PointCloud2, f'/cup_perception/{name}', 2)
            for name in ('cropped', 'above_table', 'cup_cluster')}
        self.create_subscription(CameraInfo, p('camera_info_topic',
                                               '/camera/camera/aligned_depth_to_color/camera_info'
                                               ).value,
                                 self.on_info, qos_profile_sensor_data)
        self.depth_topic = p('depth_topic',
                             '/camera/camera/aligned_depth_to_color/image_raw').value
        self.create_subscription(Image, self.depth_topic, self.on_depth, qos_profile_sensor_data)
        # Wall-clock timer, so it also fires when no simulation clock is running.
        self.create_timer(5.0, self.check_input, clock=rclpy.clock.Clock())

    def check_input(self):
        if self.last_depth is None or time.monotonic() - self.last_depth > 5.0:
            self.get_logger().warn(
                f'no depth images on {self.depth_topic} for 5 s: is the simulation '
                '(or the camera driver) running?')

    def on_info(self, msg):
        self.camera_info = msg

    def on_depth(self, msg):
        self.last_depth = time.monotonic()
        if self.camera_info is None or time.monotonic() - self.last_run < self.period:
            return
        self.last_run = time.monotonic()
        try:
            self.process(msg)
        except (ValueError, tf2_ros.TransformException) as e:
            self.get_logger().warn(f'skipping frame: {e}', throttle_duration_sec=5.0)

    def publish(self, name, points, stamp):
        header = Header(stamp=stamp, frame_id=self.world)
        self.pubs[name].publish(point_cloud2.create_cloud_xyz32(header, points.astype(np.float32)))

    def process(self, msg):
        t0 = time.monotonic()
        # 1. Depth -> points in the camera frame -> world frame.
        depth = cloud_ops.depth_to_metres(msg.data, msg.encoding, msg.height, msg.width, msg.step)
        k = self.camera_info.k
        pts_cam = cloud_ops.depth_to_points(depth, k[0], k[4], k[2], k[5], self.stride,
                                            *self.range)
        tf = self.tf_buffer.lookup_transform(self.world, msg.header.frame_id, msg.header.stamp,
                                             timeout=Duration(seconds=0.2))
        pts = cloud_ops.transform(pts_cam, matrix_from_transform(tf.transform))

        # 2. Crop to the box above the table.
        cropped = cloud_ops.crop_box(pts, *self.crop)
        self.publish('cropped', cropped, msg.header.stamp)

        # 3. Table plane and the points above it.
        plane = cloud_ops.fit_table_plane(cropped, *self.plane_cfg)
        if plane is None:
            self.get_logger().warn('no table plane found', throttle_duration_sec=5.0)
            return
        h = plane.height_of(cropped)
        above = cropped[h > self.min_above]
        self.publish('above_table', above, msg.header.stamp)

        # 4. Cluster and pick the cup.
        labels = cloud_ops.dbscan(above, *self.cluster_cfg)
        clusters = cloud_ops.describe_clusters(above, labels, plane)
        cup, rejected = cloud_ops.select_cup(clusters, *self.cup_cfg)
        self.publish('cup_cluster', cup.points if cup else np.zeros((0, 3)), msg.header.stamp)

        tilt = math.degrees(math.acos(min(1.0, plane.normal[2])))
        status = (f'{len(pts)} pts -> cropped {len(cropped)} -> table z '
                  f'{plane.z_at(0.5, 0.0):.3f} (tilt {tilt:.1f} deg, {plane.inliers} inliers) '
                  f'-> above {len(above)} -> {len(clusters)} clusters -> ')
        if cup is None:
            status += 'no cup-sized cluster' + (
                f' (rejected {len(rejected)}: {"; ".join(rejected[:3])})' if rejected else '')
        else:
            c = cup.points.mean(axis=0)
            status += (f'cup cluster: {len(cup.points)} pts, {cup.height * 1000:.0f} mm tall, '
                       f'{cup.footprint * 1000:.0f} mm wide, centroid '
                       f'({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})')
        self.get_logger().info(f'{status} [{(time.monotonic() - t0) * 1000:.0f} ms]',
                               throttle_duration_sec=2.0)


def main():
    rclpy.init()
    rclpy.spin(CupPoseEstimator())


if __name__ == '__main__':
    main()
